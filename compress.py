import fitz  # PyMuPDF
import os

def aggressive_compress(input_path, output_path, dpi=120, quality=60):
    if not os.path.exists(input_path):
        print("Source file '1.pdf' not found.")
        return

    doc = fitz.open(input_path)
    pdf_writer = fitz.open()  # Create new PDF

    for page in doc:
        # 1. Render page to an image
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
        
        # 2. Compress that image to JPEG bytes
        img_bytes = pix.tobytes("jpg", jpg_quality=quality)
        
        # 3. Create a blank page in the new PDF with the same dimensions
        new_page = pdf_writer.new_page(width=page.rect.width, 
                                      height=page.rect.height)
        
        # 4. Insert the compressed image to fill the new page
        new_page.insert_image(new_page.rect, stream=img_bytes)

    # Save with maximum structural optimization
    pdf_writer.save(output_path, garbage=4, deflate=True)
    pdf_writer.close()
    doc.close()
    
    final_size = os.path.getsize(output_path) / 1024
    print(f"Success! '{output_path}' is {final_size:.2f} KB")

aggressive_compress("2.pdf", "UQ畢業證書（認證）.pdf", dpi=100, quality=50)