import json

from docling.document_converter import DocumentConverter


def explore_docling(file_path):
    print(f"--- Running Docling on {file_path} ---")
    converter = DocumentConverter()

    # Convert the document (works for PDF and DOCX)
    result = converter.convert(file_path)

    # Export the parsed document structure to a dictionary
    doc_dict = result.document.export_to_dict()

    # Print the top-level keys to understand the intermediate representation (IR)
    print("Top-level structure keys:", doc_dict.keys())

    # Inspect the first few texts/items
    print("\nSample extracted items:")
    for item in doc_dict.get("texts", []):
        print(f"Type: {item.get('label')} | Content: {item.get('text')}")

    # Optional: Dump full JSON to a file for deep inspection
    out_file = f"{file_path}_docling_ir.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, indent=2)
    print(f"\nFull Docling IR saved to {out_file}")


if __name__ == "__main__":
    explore_docling("sample.pdf")  # Try with a digital-born PDF or DOCX
