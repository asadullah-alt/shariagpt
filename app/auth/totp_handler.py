"""
TOTP Handler — Time-based One-Time Password logic (2FA)
"""
import pyotp
import qrcode
import base64
from io import BytesIO

def generate_totp_secret() -> str:
    """Generate a random base32 TOTP secret."""
    return pyotp.random_base32()

def get_provisioning_uri(email: str, secret: str) -> str:
    """Get the otpauth:// URI for QR code generation."""
    return pyotp.totp.TOTP(secret).provisioning_uri(
        name=email,
        issuer_name="ShariaGPT"
    )

def generate_qr_base64(uri: str) -> str:
    """Generate a base64 encoded PNG of the QR code for frontend rendering."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(uri)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def verify_totp(secret: str, code: str) -> bool:
    """Verify a TOTP code against the secret."""
    totp = pyotp.TOTP(secret)
    return totp.verify(code)
