import numpy as np

def BGRtoYCrCb(inputImage):
  result = np.zeros(inputImage.shape)
  result = result.astype(np.float64)

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

DCT_matrix = np.array([[ 0.35355339,  0.35355339,  0.35355339,  0.35355339,  0.35355339,  0.35355339,
   0.35355339,  0.35355339],
 [ 0.49039264,  0.41573481,  0.27778512,  0.09754516, -0.09754516, -0.27778512,
  -0.41573481, -0.49039264],
 [ 0.46193977,  0.19134172, -0.19134172, -0.46193977, -0.46193977, -0.19134172,
   0.19134172,  0.46193977],
 [ 0.41573481, -0.09754516, -0.49039264, -0.27778512,  0.27778512,  0.49039264,
   0.09754516, -0.41573481],
 [ 0.35355339, -0.35355339, -0.35355339,  0.35355339,  0.35355339, -0.35355339,
  -0.35355339,  0.35355339],
 [ 0.27778512, -0.49039264,  0.09754516,  0.41573481, -0.41573481, -0.09754516,
   0.49039264, -0.27778512],
 [ 0.19134172, -0.46193977,  0.46193977, -0.19134172, -0.19134172,  0.46193977,
  -0.46193977,  0.19134172],
 [ 0.09754516, -0.27778512,  0.41573481, -0.49039264,  0.49039264, -0.41573481,
   0.27778512, -0.09754516]], dtype = np.float64)

DCT_T_matrix = DCT_matrix.T

def calc_dct(f):
  return np.dot(np.dot(DCT_matrix, f), DCT_T_matrix)


zigzagOrder = np.array([0,1,8,16,9,2,3,10,17,24,32,25,18,11,4,5,12,19,26,33,40,48,41,34,27,20,13,6,7,14,21,28,35,42,
                           49,56,57,50,43,36,29,22,15,23,30,37,44,51,58,59,52,45,38,31,39,46,53,60,61,54,47,55,62,63])

def zig_zag(matrix):
  matrix = matrix.flatten()
  return matrix[zigzagOrder]

def preprocess(img):
  last_dc = 0
  dc_size_list = []
  dc_vli_list = []
  ac_first_byte_list = []
  ac_huffman_list = []
  ac_vli_list = []
  for j in range(0, img.shape[0], 8):
    for k in range(0, img.shape[1], 8):
      block_dct_zig_zag = zig_zag(img[j:j+8, k:k+8])
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


def variable_length_int_encode(num):
    if num == 0:
        return ''
    elif num > 0:
        return bin(int(num))[2:]
    elif num < 0:
        bits = bin(abs(int(num)))[2:]
        return ''.join(map(lambda c: '0' if c == '1' else '1', bits))