#!/usr/bin/env python3
"""
Bitcoin Private Key to WIF Converter with Address Verification
Converts hex private key to WIF format and shows all derived addresses.
"""

import hashlib
import sys

# Base58 alphabet
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def sha256(data):
    return hashlib.sha256(data).digest()

def ripemd160(data):
    h = hashlib.new('ripemd160')
    h.update(data)
    return h.digest()

def hash160(data):
    return ripemd160(sha256(data))

def double_sha256(data):
    return sha256(sha256(data))

def base58_encode(data):
    num = int.from_bytes(data, 'big')
    result = ''
    while num > 0:
        num, remainder = divmod(num, 58)
        result = BASE58_ALPHABET[remainder] + result
    # Handle leading zeros
    for byte in data:
        if byte == 0:
            result = '1' + result
        else:
            break
    return result

def base58check_encode(version, payload):
    data = bytes([version]) + payload
    checksum = double_sha256(data)[:4]
    return base58_encode(data + checksum)

def bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = chk >> 25
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            chk ^= GEN[i] if ((b >> i) & 1) else 0
    return chk

def bech32_hrp_expand(hrp):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]

def bech32_create_checksum(hrp, data):
    values = bech32_hrp_expand(hrp) + data
    polymod = bech32_polymod(values + [0,0,0,0,0,0]) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]

def bech32_encode(hrp, witver, witprog):
    CHARSET = 'qpzry9x8gf2tvdw0s3jn54khce6mua7l'
    data = [witver] + convertbits(witprog, 8, 5)
    combined = data + bech32_create_checksum(hrp, data)
    return hrp + '1' + ''.join([CHARSET[d] for d in combined])

def convertbits(data, frombits, tobits, pad=True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << tobits) - 1
    for value in data:
        acc = (acc << frombits) | value
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    return ret

def get_public_key(private_key_int):
    """Derive public key from private key using secp256k1"""
    # secp256k1 parameters
    P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    
    def modinv(a, m):
        if a < 0:
            a = a % m
        g, x, _ = extended_gcd(a, m)
        if g != 1:
            raise Exception('Modular inverse does not exist')
        return x % m
    
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    def point_add(p1, p2):
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2 and y1 != y2:
            return None
        if x1 == x2:
            m = (3 * x1 * x1) * modinv(2 * y1, P) % P
        else:
            m = (y2 - y1) * modinv(x2 - x1, P) % P
        x3 = (m * m - x1 - x2) % P
        y3 = (m * (x1 - x3) - y1) % P
        return (x3, y3)
    
    def scalar_mult(k, point):
        result = None
        addend = point
        while k:
            if k & 1:
                result = point_add(result, addend)
            addend = point_add(addend, addend)
            k >>= 1
        return result
    
    pub = scalar_mult(private_key_int, (Gx, Gy))
    return pub

def hex_to_wif(hex_key):
    # Clean up input
    hex_key = hex_key.strip().lower()
    if hex_key.startswith('0x'):
        hex_key = hex_key[2:]
    if hex_key.startswith('-'):
        print("ERROR: Negative key detected! This is invalid.")
        return None
    
    # Pad to 64 characters (32 bytes)
    hex_key = hex_key.zfill(64)
    
    try:
        private_key_bytes = bytes.fromhex(hex_key)
        private_key_int = int(hex_key, 16)
    except ValueError:
        print(f"ERROR: Invalid hex string: {hex_key}")
        return None
    
    # Get public key
    pub = get_public_key(private_key_int)
    if pub is None:
        print("ERROR: Invalid private key")
        return None
    
    pub_x, pub_y = pub
    
    # Compressed public key (33 bytes)
    prefix = b'\x02' if pub_y % 2 == 0 else b'\x03'
    pub_compressed = prefix + pub_x.to_bytes(32, 'big')
    
    # Uncompressed public key (65 bytes)
    pub_uncompressed = b'\x04' + pub_x.to_bytes(32, 'big') + pub_y.to_bytes(32, 'big')
    
    # WIF Compressed (add 0x01 suffix before checksum)
    wif_compressed = base58check_encode(0x80, private_key_bytes + b'\x01')
    
    # WIF Uncompressed
    wif_uncompressed = base58check_encode(0x80, private_key_bytes)
    
    # Addresses from compressed public key
    h160_compressed = hash160(pub_compressed)
    addr_p2pkh_compressed = base58check_encode(0x00, h160_compressed)  # Legacy 1...
    addr_p2sh_compressed = base58check_encode(0x05, hash160(b'\x00\x14' + h160_compressed))  # P2SH-wrapped 3...
    addr_bech32_compressed = bech32_encode('bc', 0, list(h160_compressed))  # Native SegWit bc1q...
    
    # Addresses from uncompressed public key  
    h160_uncompressed = hash160(pub_uncompressed)
    addr_p2pkh_uncompressed = base58check_encode(0x00, h160_uncompressed)  # Legacy 1...
    
    return {
        'hex': '0x' + hex_key.upper().lstrip('0'),
        'wif_compressed': wif_compressed,
        'wif_uncompressed': wif_uncompressed,
        'pubkey_compressed': pub_compressed.hex(),
        'pubkey_uncompressed': pub_uncompressed.hex(),
        'addr_legacy_compressed': addr_p2pkh_compressed,
        'addr_segwit_p2sh': addr_p2sh_compressed,
        'addr_segwit_native': addr_bech32_compressed,
        'addr_legacy_uncompressed': addr_p2pkh_uncompressed,
    }

def main():
    print("=" * 70)
    print("Bitcoin Private Key to WIF Converter (with Address Verification)")
    print("=" * 70)
    
    if len(sys.argv) > 1:
        hex_key = sys.argv[1]
    else:
        # Try to read from KEYFOUND.txt
        try:
            with open('KEYFOUND.txt', 'r') as f:
                content = f.read()
                # Find hex key in file
                import re
                match = re.search(r'Priv:\s*(0x[0-9A-Fa-f]+)', content)
                if match:
                    hex_key = match.group(1)
                    print(f"\nRead from KEYFOUND.txt: {hex_key}")
                else:
                    print("\nUsage: python3 hex_to_wif.py <hex_private_key>")
                    print("   or: python3 hex_to_wif.py   (reads from KEYFOUND.txt)")
                    return
        except FileNotFoundError:
            print("\nUsage: python3 hex_to_wif.py <hex_private_key>")
            print("   or: python3 hex_to_wif.py   (reads from KEYFOUND.txt)")
            return
    
    result = hex_to_wif(hex_key)
    if result is None:
        return
    
    print(f"\nInput (Hex):  {result['hex']}")
    print("-" * 70)
    
    print("\n📍 ADDRESSES (use these to verify which format your wallet needs):\n")
    
    print("   From COMPRESSED public key (Bitcoin Puzzles use this!):")
    print(f"   • Legacy (P2PKH):     {result['addr_legacy_compressed']}")
    print(f"   • SegWit (P2SH):      {result['addr_segwit_p2sh']}")
    print(f"   • SegWit (Native):    {result['addr_segwit_native']}")
    
    print("\n   From UNCOMPRESSED public key:")
    print(f"   • Legacy (P2PKH):     {result['addr_legacy_uncompressed']}")
    
    print("\n" + "-" * 70)
    print("\n🔑 WIF PRIVATE KEYS:\n")
    print(f"   Compressed:   {result['wif_compressed']}")
    print(f"   Uncompressed: {result['wif_uncompressed']}")
    
    print("\n" + "-" * 70)
    print("\n📋 PUBLIC KEYS:\n")
    print(f"   Compressed:   {result['pubkey_compressed']}")
    print(f"   Uncompressed: {result['pubkey_uncompressed'][:66]}...")
    
    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT: Bitcoin Puzzle addresses start with '1' and use")
    print("   COMPRESSED public keys. Import the COMPRESSED WIF (starts with K or L).")
    print("   Verify the 'Legacy (P2PKH)' address matches the puzzle address!")
    print("=" * 70)
    
    # Save to file
    with open('KEYFOUND_WIF.txt', 'w') as f:
        f.write(f"Private Key (Hex): {result['hex']}\n\n")
        f.write("WIF Keys:\n")
        f.write(f"  Compressed:   {result['wif_compressed']}\n")
        f.write(f"  Uncompressed: {result['wif_uncompressed']}\n\n")
        f.write("Addresses (Compressed):\n")
        f.write(f"  Legacy:       {result['addr_legacy_compressed']}\n")
        f.write(f"  SegWit P2SH:  {result['addr_segwit_p2sh']}\n")
        f.write(f"  SegWit Native:{result['addr_segwit_native']}\n\n")
        f.write("Address (Uncompressed):\n")
        f.write(f"  Legacy:       {result['addr_legacy_uncompressed']}\n")
    
    print(f"\nSaved to KEYFOUND_WIF.txt")

if __name__ == "__main__":
    main()
