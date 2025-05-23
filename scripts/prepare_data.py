# pipeline

from src.chunker import chunk_pdf

def run_pipeline(pdf_path):
    # Task #1: Chunk the PDF
    chunks = chunk_pdf(pdf_path)
    print('PDF was chunked...')
    
if __name__ == "__main__":
    run_pipeline("./data/dr_voss_diary.pdf")