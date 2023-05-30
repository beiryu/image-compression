from upload_app.helpers.utils.file_writer import *
from upload_app.helpers.utils.utils import *
from upload_app.helpers.utils.huffman import *


def color_encoder(file_name, img, real_height, real_width, quality):
  quan_table_lum = setup_quan_table(basic_quan_table_lum, quality)
  quan_table_chroma = setup_quan_table(basic_quan_table_chroma, quality)

  img_ycrcb = BGRtoYCrCb(img).astype(np.float64)

  img_ycrcb = img_ycrcb - 128
  for channel in range(img.shape[2]):
    if channel == 0:
      Q = quan_table_lum
    else:
      Q = quan_table_chroma
    for j in range(0, img.shape[0], 8):
      for k in range(0, img.shape[1], 8):
          img_ycrcb[j:j+8, k:k+8, channel] = calc_dct(img_ycrcb[j:j+8, k:k+8, channel])
          img_ycrcb[j:j+8, k:k+8, channel] = np.round(img_ycrcb[j:j+8, k:k+8, channel] / Q)

  dc_y_size_list, dc_y_vli_list, ac_y_first_byte_list, ac_y_huffman_list, ac_y_vli_list = preprocess(img_ycrcb[:,:,0])
  dc_cr_size_list, dc_cr_vli_list, ac_cr_first_byte_list, ac_cr_huffman_list, ac_cr_vli_list = preprocess(img_ycrcb[:,:,1])
  dc_cb_size_list, dc_cb_vli_list, ac_cb_first_byte_list, ac_cb_huffman_list, ac_cb_vli_list = preprocess(img_ycrcb[:,:,2])

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
  for i in range(len(ac_y_first_byte_list)):
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