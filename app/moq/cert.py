"""Chung chi tu ky cho WebTransport, kem ma bam de trinh duyet chap nhan.

VI SAO CAN: WebTransport bat buoc chay tren HTTPS/QUIC, ma he nay cam o mang
khach hang bang dia chi IP, khong co ten mien nen khong xin duoc chung chi
that. Duong thoat chinh thuc la `serverCertificateHashes`: trinh duyet bo qua
chuoi tin cay va chi so SHA-256 cua chung chi voi con so ta dua cho no.

Chrome dat DIEU KIEN cho duong nay, ca ba deu bat buoc:
    * khoa ECDSA duong P-256 (RSA bi tu choi thang),
    * tong thoi han hieu luc KHONG qua 14 ngay,
    * client phai truyen dung ma bam SHA-256 cua ban DER.

Nen chung chi phai TU XOAY VONG. Ta sinh moi 13 ngay va ghi ra dia de engine
khoi phai bat tay lai moi lan khoi dong lai tien trinh; het han thi sinh moi.
Trinh phat luon hoi /moq/info truoc khi ket noi nen no lay ma bam moi ngay,
khong bao gio cam ma bam cu.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import ipaddress
import os
import socket
from pathlib import Path
from typing import List, Tuple

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

# 13 ngay: duoi tran 14 ngay cua Chrome, con chua mot ngay du phong cho lech
# dong ho giua board va may xem (board nay chay khong co RTC pin nen sau khi
# mat dien gio co the lech).
VALID_DAYS = 13
# Sinh lai khi vong doi con lai duoi nguong nay.
RENEW_BEFORE_HOURS = 24


def _local_ips() -> List[str]:
    """Moi dia chi IPv4 cua board, de nhet vao SAN."""
    out = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            out.add(info[4][0])
    except OSError:
        pass
    # Cach chac an hon getaddrinfo (hostname hay tro ve 127.0.1.1): hoi kernel
    # xem di ra ngoai bang dia chi nao.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        out.add(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()
    return sorted(out)


def _build(paths: Tuple[Path, Path]) -> None:
    cert_path, key_path = paths
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "vms-moq")])

    alt: List[x509.GeneralName] = [x509.DNSName("localhost")]
    for ip in _local_ips():
        try:
            alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        # Lui mot gio: neu dong ho may xem nhanh hon board mot chut thi chung
        # chi "chua co hieu luc" va trinh duyet tu choi ma khong noi vi sao.
        .not_valid_before(now - dt.timedelta(hours=1))
        .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    os.chmod(key_path, 0o600)


def ensure(directory: str) -> dict:
    """Tra ve {cert_file, key_file, fingerprint(hex), expires_at(iso)}.

    Sinh moi neu chua co, hong, hoac sap het han.
    """
    base = Path(directory)
    cert_path, key_path = base / "moq-cert.pem", base / "moq-key.pem"

    need = True
    if cert_path.exists() and key_path.exists():
        try:
            cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
            left = cert.not_valid_after_utc - dt.datetime.now(dt.timezone.utc)
            need = left < dt.timedelta(hours=RENEW_BEFORE_HOURS)
        except Exception:
            need = True
    if need:
        _build((cert_path, key_path))

    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    der = cert.public_bytes(serialization.Encoding.DER)
    return {
        "cert_file": str(cert_path),
        "key_file": str(key_path),
        "fingerprint": hashlib.sha256(der).hexdigest(),
        "expires_at": cert.not_valid_after_utc.isoformat(),
    }
