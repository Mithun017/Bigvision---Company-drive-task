from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Master Interview Guide: Basketball Player Detection & Tracking', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(input_file, output_file):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('## '):
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, line[3:], 0, 1)
            pdf.ln(2)
        elif line.startswith('### '):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, line[4:], 0, 1)
            pdf.ln(1)
        elif line.startswith('---'):
            pdf.ln(5)
        elif line:
            pdf.set_font('Arial', '', 11)
            # handle bolding in text for fpdf (simplistic approach, just write the line)
            # replace markdown bold ** with nothing for simplicity in this basic script
            line = line.replace('**', '')
            pdf.multi_cell(0, 6, line)
            pdf.ln(2)

    pdf.output(output_file)

if __name__ == '__main__':
    create_pdf('master_explanation.txt', 'Master_Interview_Guide.pdf')
