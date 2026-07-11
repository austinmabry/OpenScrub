#!/usr/bin/env python3
"""openscrub_vault.py — password-based at-rest encryption for the job store.

The job store (uploads, normalized/intermediate video, audit reports,
detection thumbnails) contains PHI in plaintext. This module lets the web
app encrypt all of it with a key derived from a user-chosen password:

  * scrypt (n=2^15, r=8, p=1) derives a key-encryption key from the
    password; a random 256-bit DATA key actually encrypts files, so the
    password can be changed without re-encrypting everything.
  * Files are encrypted in 4 MiB chunks with AES-256-GCM (each chunk
    authenticated; any tampering fails decryption loudly).
  * The keystore (salt + wrapped data key) lives next to the jobs dir.

Semantics (deliberately simple, v1):
  LOCKED   = every file in the job store is encrypted on disk. The server
             starts locked and refuses job operations until unlocked.
  UNLOCKED = files are decrypted in place so the whole pipeline (cv2,
             ffmpeg, review UI) works unchanged. Locking re-encrypts.
The vault re-locks on clean server shutdown. A crash while unlocked can
leave plaintext on disk — pair this with OS-level disk encryption
(BitLocker etc.) for a real at-rest guarantee.

THERE IS NO PASSWORD RESET. Losing the password makes encrypted files
permanently unrecoverable — that is the entire point of the feature.
"""

import base64
import json
import os
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"OSVAULT1"
SUFFIX = ".osvault"
CHUNK = 4 * 1024 * 1024
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 15, 8, 1


def _kdf(password, salt):
    return Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R,
                  p=SCRYPT_P).derive(password.encode("utf-8"))


# ---------------------------------------------------------------------------
# Keystore: salt + data key wrapped (AES-GCM) by the password-derived key
# ---------------------------------------------------------------------------

def keystore_path(root):
    return os.path.join(root, "vault_keystore.json")


def keystore_exists(root):
    return os.path.exists(keystore_path(root))


def create_keystore(root, password):
    """Set a password. Returns the raw data key. Refuses to overwrite."""
    path = keystore_path(root)
    if os.path.exists(path):
        raise RuntimeError("vault keystore already exists")
    salt = os.urandom(16)
    kek = _kdf(password, salt)
    data_key = os.urandom(32)
    nonce = os.urandom(12)
    wrapped = AESGCM(kek).encrypt(nonce, data_key, MAGIC)
    doc = {"version": 1, "kdf": "scrypt",
           "n": SCRYPT_N, "r": SCRYPT_R, "p": SCRYPT_P,
           "salt": base64.b64encode(salt).decode(),
           "nonce": base64.b64encode(nonce).decode(),
           "wrapped_key": base64.b64encode(wrapped).decode()}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    os.replace(tmp, path)
    return data_key


def open_keystore(root, password):
    """Unwrap the data key. Raises ValueError on a wrong password."""
    with open(keystore_path(root), encoding="utf-8") as f:
        doc = json.load(f)
    salt = base64.b64decode(doc["salt"])
    kek = _kdf(password, salt)
    try:
        return AESGCM(kek).decrypt(base64.b64decode(doc["nonce"]),
                                   base64.b64decode(doc["wrapped_key"]),
                                   MAGIC)
    except Exception:
        raise ValueError("wrong password")


# ---------------------------------------------------------------------------
# File format: MAGIC | 12-byte nonce base | [4-byte len | GCM(chunk)]...
# Chunk nonces = nonce_base XOR counter, so a nonce never repeats per key.
# ---------------------------------------------------------------------------

def _chunk_nonce(base, counter):
    c = struct.pack(">Q", counter)
    return base[:4] + bytes(a ^ b for a, b in zip(base[4:], c))


def is_encrypted(path):
    return path.endswith(SUFFIX)


def encrypt_file(key, path):
    """path -> path+SUFFIX (atomic; original removed only on success)."""
    if is_encrypted(path):
        return path
    aes = AESGCM(key)
    base = os.urandom(12)
    out = path + SUFFIX
    tmp = out + ".tmp"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(MAGIC + base)
        counter = 0
        while True:
            chunk = src.read(CHUNK)
            if not chunk:
                break
            sealed = aes.encrypt(_chunk_nonce(base, counter), chunk, None)
            dst.write(struct.pack(">I", len(sealed)) + sealed)
            counter += 1
    os.replace(tmp, out)
    os.remove(path)
    return out


def decrypt_file(key, path):
    """path+SUFFIX -> path (atomic; ciphertext removed only on success).
    Raises on tampering (GCM auth) or wrong key."""
    if not is_encrypted(path):
        return path
    aes = AESGCM(key)
    out = path[:-len(SUFFIX)]
    tmp = out + ".tmp"
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        head = src.read(len(MAGIC) + 12)
        if not head.startswith(MAGIC):
            raise ValueError("not an OpenScrub vault file: %s" % path)
        base = head[len(MAGIC):]
        counter = 0
        while True:
            ln = src.read(4)
            if not ln:
                break
            sealed = src.read(struct.unpack(">I", ln)[0])
            dst.write(aes.decrypt(_chunk_nonce(base, counter), sealed, None))
            counter += 1
    os.replace(tmp, out)
    os.remove(path)
    return out


def _walk_files(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            yield os.path.join(dirpath, f)


def encrypt_tree(key, root):
    """Encrypt every not-yet-encrypted file under root. Returns count."""
    n = 0
    for p in list(_walk_files(root)):
        if p.endswith(".tmp"):
            os.remove(p)                       # leftover from a crash
        elif not is_encrypted(p):
            encrypt_file(key, p)
            n += 1
    return n


def decrypt_tree(key, root):
    """Decrypt every encrypted file under root. Returns count."""
    n = 0
    for p in list(_walk_files(root)):
        if p.endswith(".tmp"):
            os.remove(p)
        elif is_encrypted(p):
            decrypt_file(key, p)
            n += 1
    return n


def tree_locked_state(root):
    """-> (n_encrypted, n_plain) under root (ignores .tmp leftovers)."""
    enc = plain = 0
    for p in _walk_files(root):
        if p.endswith(".tmp"):
            continue
        if is_encrypted(p):
            enc += 1
        else:
            plain += 1
    return enc, plain
