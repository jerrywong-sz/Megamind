
import torch

from src.models import build_efficientnet_b0


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_efficientnet_b0(
        pretrained=True
    ).to(device)

    # Four fake RGB images, each 224 × 224.
    images = torch.randn(
        4,
        3,
        224,
        224,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)

    print("Device:", device)
    print("Input shape:", images.shape)
    print("Output shape:", logits.shape)
    print("Raw logits:")
    print(logits)

    assert logits.shape == (4, 1), (
        f"Expected output shape (4, 1), "
        f"but received {logits.shape}"
    )

    assert torch.isfinite(logits).all(), (
        "The model produced NaN or infinite values."
    )

    print("✅ Model test passed!")


if __name__ == "__main__":
    main()
