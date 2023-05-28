import io
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from PIL import Image
import numpy as np
import base64, os, cv2, time

from upload_app.helpers.jpeg_encoder import jpeg_encoder

quality = 80
base_dir = os.path.dirname(__file__)
zigzagOrder = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])
basic_quan_table_lum = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                 [12, 12, 14, 19, 26, 58, 60, 55],
                                 [14, 13, 16, 24, 40, 57, 69, 56],
                                 [14, 17, 22, 29, 51, 87, 80, 62],
                                 [18, 22, 37, 56, 68, 109, 103, 77],
                                 [24, 35, 55, 64, 81, 104, 113, 92],
                                 [49, 64, 78, 87, 103, 121, 120, 101],
                                 [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint8)

basic_quan_table_chroma = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                    [18, 21, 26, 66, 99, 99, 99, 99],
                                    [24, 26, 56, 99, 99, 99, 99, 99],
                                    [47, 66, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99],
                                    [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.uint8)


@csrf_exempt
def home(request):
    template_name = 'index.html'

    if request.method == "GET":
        return render(request, template_name)
    
    if request.method == "POST":
        uploaded_file = request.FILES['image']  

        encoded_file = base64.b64encode(uploaded_file.read()).decode('utf-8')

        filename, file_extension = os.path.splitext(uploaded_file.name)
        output_format = file_extension[1:].upper() 
    
        image = Image.open(uploaded_file)

        if image.mode == 'RGBA':
            image = image.convert('RGB')

        image_in_dir = os.path.join(base_dir, f'static/in/{filename}.pnm')
        image_out_dir = os.path.join(base_dir, f'static/out/{filename}.{output_format}')
        
        image.save(image_in_dir)
        
        img = cv2.imread(image_in_dir, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]

        start = time.time()
        jpeg_encoder(image_out_dir, img, height, width, quality)
        end = time.time()

        with open(image_out_dir, "rb") as image_file:
            compressed_file = base64.b64encode(image_file.read()).decode('utf-8')

        encoded_file_bytes = io.BytesIO(base64.b64decode(encoded_file))
        compressed_file_bytes = io.BytesIO(base64.b64decode(compressed_file))
        
        return JsonResponse({
            "image_data": encoded_file, 
            "compress_data": compressed_file,
            "image_size": len(encoded_file_bytes.getvalue()),
            "compress_size": len(compressed_file_bytes.getvalue()),
            "time": end - start
        })