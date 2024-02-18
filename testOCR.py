import io

import fitz  # PyMuPDF
from PIL import Image

import easyocr
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


pdf_file = r"C:\Users\10942\Desktop\《格商》.pdf"
doc = fitz.open(pdf_file)
import easyocr

# 创建PDF文件
pdf_path = "output.pdf"
c = canvas.Canvas(pdf_path, pagesize=letter)

# 使用EasyOCR加载模型
reader = easyocr.Reader(['ch_sim'])
for page_num in range(len(doc)):
    page = doc.load_page(page_num)
    image_list = page.get_images()
    for image_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image = Image.open(io.BytesIO(image_bytes))
        # image.save(f"page_{page_num}_image_{image_index}.png")
        # 使用EasyOCR提取文本
        result = reader.readtext(image)


        # # 设置字体样式
        # font_name = "msyh.ttf"  # 替换为你想要使用的字体文件名
        # pdfmetrics.registerFont(TTFont('msyh', font_name))
        # c.setFont("msyh", 12)

        # 提取文本并根据位置填充到PDF中
        for detection in result:
            text = detection[1]
            # 提取文本位置信息
            bbox = detection[0]
            x_min, y_min = bbox[0][0], bbox[0][1]
            x_max, y_max = bbox[2][0], bbox[2][1]
            width = x_max - x_min
            height = y_max - y_min
            # 将文本填充到PDF中
            c.drawString(x_min, letter[1] - y_min, text)

# 保存PDF文件
c.save()