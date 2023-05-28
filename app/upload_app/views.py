from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from PIL import Image
import numpy as np
import base64, os, cv2, struct, math, time

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
        color_encoder(image_out_dir, img, height, width, quality)
        end = time.time()

        with open(image_out_dir, "rb") as image_file:
            compressed_file = base64.b64encode(image_file.read()).decode('utf-8')

        return JsonResponse({
            "image_data": encoded_file, 
            "compress_data": compressed_file,
            # "image_size": len(encoded_file),
            # "compress_size": len(compressed_file),
            "time": end - start
        })
        

def BGRtoYCrCb(inputImage):
    result = np.empty_like(inputImage)
    result = result.astype(np.float32)

    B = inputImage[:,:,0]
    G = inputImage[:,:,1]
    R = inputImage[:,:,2]

    # Y
    result[:,:,0] = 0.299 * R + 0.587 * G + 0.114 * B
    # Cr
    result[:,:,1] = (R - result[:,:,0]) * 0.713 + 128
    # Cb
    result[:,:,2] = (B - result[:,:,0]) * 0.564 + 128

    return np.uint8(result)


def separateBlock(img, block_shape):
    height, width = img.shape[:2]
    block_height, block_width = block_shape
    shape = (height // block_height, width // block_width, block_height, block_width)
    strides = img.itemsize * np.array([width * block_height, block_width, width, 1])
    img_blocks = np.lib.stride_tricks.as_strided(img, shape, strides).astype('float64')
    img_blocks = np.reshape(img_blocks, (shape[0] * shape[1], block_height, block_width))
    return img_blocks


def zig_zag(matrix):
    rows, columns = matrix.shape[:2]
    matrix_zig_zag = np.zeros(rows * columns, dtype = matrix.dtype)

    for i in range(len(zigzagOrder)):
        matrix_zig_zag[i] = matrix[int(zigzagOrder[i] / columns)][int(zigzagOrder[i] % columns)]

    return matrix_zig_zag


def variable_length_int_encode(num):
    if num == 0:
        return ''
    elif num > 0:
        return bin(int(num))[2:]
    elif num < 0:
        bits = bin(abs(int(num)))[2:]
        return ''.join(map(lambda c: '0' if c == '1' else '1', bits))


def run_length_encode(array):
    last_nonzero_index = 0
    for i, num in enumerate(array[::-1]):
        if num != 0:
            last_nonzero_index = len(array) - i
            break

    run_length = 0
    first_byte_list = []
    vli_list = []
    for i, num in enumerate(array):
        if i >= last_nonzero_index:
            first_byte_list.append(0)
            vli_list.append('')
            break
        elif num == 0 and run_length < 15:
            run_length += 1
        else:
            num_bits = variable_length_int_encode(num)
            size = len(num_bits)
            first_byte = int(bin(run_length)[2:].zfill(4) + bin(size)[2:].zfill(4), 2)

            first_byte_list.append(first_byte)
            vli_list.append(num_bits)
            run_length = 0

    return first_byte_list, vli_list


def delta_encode(dc, last_dc):
    num_bits = variable_length_int_encode(dc - last_dc)
    size = len(num_bits)

    return size, num_bits


def block_preprocess(img_blocks, block_sum, quan_table):
    last_dc = 0
    dc_size_list = []
    dc_vli_list = []
    ac_first_byte_list = []
    ac_huffman_list = []
    ac_vli_list = []
    for i in range(block_sum):
        block = img_blocks[i] - 128
        block_dct = calc_dct(block)
        block_dct_quantized = np.round(block_dct / quan_table).astype(np.int32)
        block_dct_zig_zag = zig_zag(block_dct_quantized)
        dc = block_dct_zig_zag[0]
        ac = block_dct_zig_zag[1:]

        dc_size, dc_vli = delta_encode(dc, last_dc)
        ac_first_byte_block_list, ac_vli_block_list = run_length_encode(ac)

        dc_size_list.append(dc_size)
        dc_vli_list.append(dc_vli)
        ac_first_byte_list.append(ac_first_byte_block_list)
        ac_huffman_list += ac_first_byte_block_list
        ac_vli_list.append(ac_vli_block_list)

        last_dc = dc

    return dc_size_list, dc_vli_list, ac_first_byte_list, ac_huffman_list, ac_vli_list


def setup_quan_table(basic_quan_table, quality):
    if quality >= 50:
        quality = 200 - 2 * quality
    else:
        quality = 5000 / quality

    basic_quan_table = basic_quan_table.astype(np.uint32)
    quan_table = (basic_quan_table * quality + 50) / 100
    quan_table = np.clip(quan_table, 1, 255)
    quan_table = quan_table.astype(np.uint8)
    return quan_table


def calc_dct(f):
    # Define the DCT matrix
    N = len(f)
    DCT_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                DCT_matrix[i, j] = 1 / np.sqrt(N)
            else:
                DCT_matrix[i, j] = np.sqrt(2 / N) * np.cos((np.pi * (2 * j + 1) * i) / (2 * N))

    # Compute the DCT D = DCT_Matrix @ Image_Block @ DCT_Matrix.T
    return np.dot(np.dot(DCT_matrix, f), DCT_matrix.T)


class Node:
  def __init__(self):
      self.symbol = None
      self.lchild = None
      self.rchild = None
      self.freq = 0
      self.print_text = ''

  def __lt__(self, other):
      return self.freq < other.freq


class HuffmanTree:
  def raw_to_canonical(self, code_dict):
      count = 0
      last_code_len = 0
      last_code = ''
      code_dict_canonical = {}
      for symbol, code in code_dict.items():
          code_len = len(code)
          if count == 0:
              new_code = '0' * code_len
              code_dict_canonical[symbol] = new_code
              last_code = new_code
              last_code_len = code_len
              count += 1
              continue

          if code_len == last_code_len:
              new_code = bin(int(last_code, 2) + 1)[2:].zfill(code_len)
              code_dict_canonical[symbol] = new_code

          else:
              new_code = bin((int(last_code, 2) + 1) << (code_len - last_code_len))[2:].zfill(
                  code_len)
              code_dict_canonical[symbol] = new_code

          last_code = new_code
          last_code_len = code_len
          count += 1

      return code_dict_canonical

  def create_graph(self, graph, node, pos_dict=None, x=0, y=0, layer=1):
      if pos_dict is None:
          pos_dict = {}

      pos_dict[node.print_text] = (x, y)

      if node.lchild is not None:
          graph.add_edge(node.print_text, node.lchild.print_text)
          l_x, l_y = x - 1 / 2 ** layer, y - 1
          l_layer = layer + 1
          self.create_graph(graph, node.lchild, pos_dict, l_x, l_y, l_layer)

      if node.rchild is not None:
          graph.add_edge(node.print_text, node.rchild.print_text)
          r_x, r_y = x + 1 / 2 ** layer, y - 1
          r_layer = layer + 1
          self.create_graph(graph, node.rchild, pos_dict, r_x, r_y, r_layer)

      return graph, pos_dict


class HuffmanEncoder(HuffmanTree):
    def __init__(self, array):
        self.array = array
        self.freq_dict = self.calc_freq()
        self.root_node = self.build_tree()
        self.code_dict_raw = self.calc_code(self.root_node)
        self.code_dict_raw.pop('eof')
        self.code_dict_raw = dict(sorted(self.code_dict_raw.items(), key=lambda x: len(x[1])))
        self.code_dict = self.raw_to_canonical(self.code_dict_raw)

    def calc_freq(self):
        array_len = len(self.array)

        freq_dict = {}
        for symbol in self.array:
            if symbol in freq_dict:
                freq_dict[symbol] += 1
            else:
                freq_dict[symbol] = 1

        min_freq = array_len * (2 ** (-14))
        for symbol, freq in freq_dict.items():
            if freq <= min_freq:
                freq_dict[symbol] = min_freq + 1
        return freq_dict

    def build_tree(self):
        node_list = []
        for symbol in self.freq_dict:
            node = Node()
            node.symbol = symbol
            node.freq = self.freq_dict[symbol]
            node.print_text = f'{repr(node.symbol)}:{node.freq}'
            node_list.append(node)

        eof_node = Node()
        eof_node.symbol = 'eof'
        eof_node.freq = 0
        eof_node.print_text = f'eof:{0}'
        node_list.append(eof_node)

        count = 0
        while len(node_list) > 1:
            node_list.sort(reverse=True)
            root_node = Node()
            lchild = node_list.pop()
            rchild = node_list.pop()
            root_node.freq = lchild.freq + rchild.freq
            root_node.lchild = lchild
            root_node.rchild = rchild
            root_node.print_text = f'root{count}:{root_node.freq}'
            node_list.append(root_node)
            count += 1

        return node_list[0]

    def calc_code(self, node, code_dict=None, code=''):
        if code_dict is None:
            code_dict = {}

        if node.symbol is not None:
            code_dict[node.symbol] = code
            return
        code += '0'

        if node.lchild is not None:
            self.calc_code(node.lchild, code_dict, code)
        code = code[:-1]
        code += '1'

        if node.rchild is not None:
            self.calc_code(node.rchild, code_dict, code)

        return code_dict

    def encode(self, array):
        array_encoded = []

        for symbol in array:
            code = self.code_dict[symbol]
            array_encoded.append(code)

        return array_encoded


def write_soi():
    marker = b'\xff\xd8'
    return marker


def write_app0():
    marker = b'\xff\xe0'
    app0_type = b'JFIF\x00'
    version = b'\x01\x01'
    units = b'\x00'
    density = b'\x00\x01\x00\x01'
    thumbnail = b'\x00\x00'

    app0_data = app0_type + version + units + density + thumbnail
    size = struct.pack('>H', len(app0_data) + 2)

    app0 = marker + size + app0_data
    return app0


def write_dqt(quan_table, num):
    marker = b'\xff\xdb'
    if num == 0:
        quan_table_info = b'\x00'
    else:
        quan_table_info = b'\x01'
    quan_table_bytes = zig_zag(quan_table).tobytes()

    dqt_data = quan_table_info + quan_table_bytes
    size = struct.pack('>H', len(dqt_data) + 2)

    dqt = marker + size + dqt_data
    return dqt


def write_sof(height, width, channel_num):
    marker = b'\xff\xc0'
    precision = b'\x08'
    y_image = struct.pack('>H', height)
    x_image = struct.pack('>H', width)
    components_num = struct.pack('>B', channel_num)

    components = b''
    for i in range(channel_num):
        component_index = struct.pack('>B', i + 1)
        sample_factor = b'\x11'

        if i == 0:
            quan_table_index = b'\x00'
        else:
            quan_table_index = b'\x01'

        components += component_index + sample_factor + quan_table_index

    sof_data = precision + y_image + x_image + components_num + components
    size = struct.pack('>H', len(sof_data) + 2)
    sof = marker + size + sof_data
    return sof


def write_dht(code_dict, num):
    marker = b'\xff\xc4'
    if num == 0:
        huffman_table_info = b'\x00'
    elif num == 1:
        huffman_table_info = b'\x10'
    elif num == 2:
        huffman_table_info = b'\x01'
    else:
        huffman_table_info = b'\x11'

    count = 0
    length_array = np.zeros(16, dtype=np.uint8)
    symbol_array = np.zeros(len(code_dict), dtype=np.uint8)
    for symbol, code in code_dict.items():
        length_array[len(code) - 1] += 1
        symbol_array[count] = symbol
        count += 1

    length_bytes = length_array.tobytes()
    symbol_bytes = symbol_array.tobytes()

    dht_data = huffman_table_info + length_bytes + symbol_bytes
    size = struct.pack('>H', len(dht_data) + 2)
    dht = marker + size + dht_data

    return dht


def write_sos(channel_num, image_data):
    marker = b'\xff\xda'
    components_num = struct.pack('>B', channel_num)

    components = b''
    for i in range(channel_num):
        component_index = struct.pack('>B', i + 1)

        if i == 0:
            huffman_table_index = b'\x00'
        else:
            huffman_table_index = b'\x11'

        components += component_index + huffman_table_index

    end = b'\x00\x3f\x00'

    sos_data = components_num + components + end
    size = struct.pack('>H', len(sos_data) + 2)
    sos = marker + size + sos_data + image_data
    return sos


def write_eoi():
    marker = b'\xff\xd9'
    return marker


def write_jpeg(file_name, height, width, channel_num, image_data, quan_table_list, huffman_code_dict_list):
    soi = write_soi()
    app0 = write_app0()

    count = 0
    dqt = b''
    for quan_table in quan_table_list:
        dqt += write_dqt(quan_table, count)
        count += 1

    sof = write_sof(height, width, channel_num)

    count = 0
    dht = b''
    for code_dict in huffman_code_dict_list:
        dht += write_dht(code_dict, count)
        count += 1

    sos = write_sos(channel_num, image_data)
    eoi = write_eoi()

    jpeg = soi + app0 + dqt + sof + dht + sos + eoi
    f = open(file_name, 'wb')
    f.write(jpeg)
    f.close()


def color_encoder(file_name, img, real_height, real_width, quality):
    block_shape = (8, 8)
    img_ycrcb = BGRtoYCrCb(img)

    img_y, img_cr, img_cb = cv2.split(img_ycrcb)

    filled_height, filled_width = img_y.shape[:2]
    block_sum = math.ceil(filled_height // block_shape[0]) * math.ceil(filled_width // block_shape[1])
    
    img_y_blocks = separateBlock(img_y, block_shape)
    img_cr_blocks = separateBlock(img_cr, block_shape)
    img_cb_blocks = separateBlock(img_cb, block_shape)

    quan_table_lum = setup_quan_table(basic_quan_table_lum, quality)
    quan_table_chroma = setup_quan_table(basic_quan_table_chroma, quality)

    dc_y_size_list, dc_y_vli_list, ac_y_first_byte_list, ac_y_huffman_list, ac_y_vli_list = block_preprocess(
        img_y_blocks,
        block_sum,
        quan_table_lum)
    dc_cr_size_list, dc_cr_vli_list, ac_cr_first_byte_list, ac_cr_huffman_list, ac_cr_vli_list = block_preprocess(
        img_cr_blocks, block_sum, quan_table_chroma)
    dc_cb_size_list, dc_cb_vli_list, ac_cb_first_byte_list, ac_cb_huffman_list, ac_cb_vli_list = block_preprocess(
        img_cb_blocks, block_sum, quan_table_chroma)


    huffman_encoder_dc_y = HuffmanEncoder(dc_y_size_list)
    code_dict_dc_y = huffman_encoder_dc_y.code_dict
    huffman_encoder_ac_y = HuffmanEncoder(ac_y_huffman_list)
    code_dict_ac_y = huffman_encoder_ac_y.code_dict

    huffman_encoder_dc_chroma = HuffmanEncoder(dc_cr_size_list + dc_cb_size_list)
    code_dict_dc_chroma = huffman_encoder_dc_chroma.code_dict
    huffman_encoder_ac_chroma = HuffmanEncoder(ac_cr_huffman_list + ac_cb_huffman_list)
    code_dict_ac_chroma = huffman_encoder_ac_chroma.code_dict

    dc_y_size_list_encoded = huffman_encoder_dc_y.encode(dc_y_size_list)
    dc_cr_size_list_encoded = huffman_encoder_dc_chroma.encode(dc_cr_size_list)
    dc_cb_size_list_encoded = huffman_encoder_dc_chroma.encode(dc_cb_size_list)

    image_data_bits = ''
    for i in range(block_sum):
        ac_y_first_byte_encoded = huffman_encoder_ac_y.encode(ac_y_first_byte_list[i])
        ac_cr_first_byte_encoded = huffman_encoder_ac_chroma.encode(ac_cr_first_byte_list[i])
        ac_cb_first_byte_encoded = huffman_encoder_ac_chroma.encode(ac_cb_first_byte_list[i])

        block_encoded = dc_y_size_list_encoded[i] + dc_y_vli_list[i]
        for j in range(len(ac_y_first_byte_encoded)):
            block_encoded += ac_y_first_byte_encoded[j] + ac_y_vli_list[i][j]

        block_encoded += dc_cb_size_list_encoded[i] + dc_cb_vli_list[i]
        for j in range(len(ac_cb_first_byte_encoded)):
            block_encoded += ac_cb_first_byte_encoded[j] + ac_cb_vli_list[i][j]

        block_encoded += dc_cr_size_list_encoded[i] + dc_cr_vli_list[i]
        for j in range(len(ac_cr_first_byte_encoded)):
            block_encoded += ac_cr_first_byte_encoded[j] + ac_cr_vli_list[i][j]

        image_data_bits += block_encoded

    if len(image_data_bits) % 8 != 0:
        image_data_bits += (8 - (len(image_data_bits) % 8)) * '1'

    image_data = int(image_data_bits, 2).to_bytes(len(image_data_bits) // 8, 'big')
    image_data = image_data.replace(b'\xff', b'\xff\x00')

    write_jpeg(file_name, real_height, real_width, 3, image_data, [quan_table_lum, quan_table_chroma],
                [code_dict_dc_y, code_dict_ac_y, code_dict_dc_chroma, code_dict_ac_chroma])
