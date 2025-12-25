import argparse
import json
import os
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser("Vehicle Detection & Classification CLI")
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--backend",
        choices=["openvino", "pytorch", "compare"],
        default="openvino",
    )
    parser.add_argument("--device", default="CPU")

    # 👇 VLM Q/A
    parser.add_argument(
        "--question",
        help="Ask a question about the image (VLM Q/A)",
    )

    args = parser.parse_args()
    os.makedirs("outputs", exist_ok=True)

    result = run_pipeline(
        image_path=args.image,
        backend=args.backend,
        device=args.device,
        qa_question=args.question,
    )

    # ---- JSON (metrics only) ----
    if args.backend == "compare":
        print(json.dumps({
            "pytorch": {
                "latency_ms": result["compare"]["pytorch"]["latency_ms"],
                "fps": result["compare"]["pytorch"]["fps"],
                "vehicles": result["compare"]["pytorch"]["vehicles"],
            },
            "openvino": {
                "latency_ms": result["compare"]["openvino"]["latency_ms"],
                "fps": result["compare"]["openvino"]["fps"],
                "vehicles": result["compare"]["openvino"]["vehicles"],
            }
        }, indent=2))
    else:
        print(json.dumps({
            "backend": result["backend"],
            "latency_ms": result["latency_ms"],
            "fps": result["fps"],
            "vehicles": result["vehicles"],
        }, indent=2))

    # ---- VLM Q/A (HUMAN OUTPUT) ----
    if args.question:
        print("\n🧠 Visual Question Answering")
        print(f"Q: {args.question}")
        print(f"A: {result.get('qa_answer')}")

    print("\n✅ Output image saved in outputs/")


if __name__ == "__main__":
    main()
