import fitz  # PyMuPDF


def explore_pymupdf_geometry(pdf_path):
    print(f"--- Running PyMuPDF Geometry Extraction on {pdf_path} ---")
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):  # Inspecting first 2 pages
        page = doc[page_num]
        print(f"\nPage {page_num + 1} Dimensions: {page.rect}")

        # Extract text as a dictionary (preserves blocks, lines, spans, BBOXes)
        page_dict = page.get_text("dict")
        blocks = page_dict.get("blocks", [])

        print(f"Found {len(blocks)} blocks on page {page_num + 1}")

        for b_idx, block in enumerate(blocks):  # Look at the first 3 blocks
            if block['type'] == 0:  # 0 means text block (1 is image)
                bbox = block['bbox']
                print(f"\nBlock {b_idx} BBOX: {bbox}")

                # Inspect the first line in the block
                if block['lines']:
                    first_line = block['lines'][0]
                    line_bbox = first_line['bbox']
                    # Spans contain the actual text and font styling
                    text = "".join([span['text'] for span in first_line['spans']])
                    print(f"  └─ Line 1 BBOX: {line_bbox} | Text: '{text.strip()}'")


if __name__ == "__main__":
    explore_pymupdf_geometry("sample.pdf")
