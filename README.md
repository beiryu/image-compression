# Nén ảnh theo chuẩn JPEG

## 1. Mô tả về thuật toán nén ảnh JPEG

JPEG là một thuật toán nén hình ảnh được sử dụng rộng rãi trên các thiết bị và
ứng dụng điện tử. Thuật toán này sử dụng phương pháp nén mất mát, có nghĩa
là một số thông tin hình ảnh sẽ bị mất đi sau khi nén.

Input của thuật toán JPEG là một hình ảnh kỹ thuật số (được biểu diễn dưới
dạng ma trận các giá trị pixel).

Output của thuật toán là một tệp hình ảnh đã được nén và được lưu trữ dưới
dạng một tệp tin hình ảnh (ví dụ: JPEG, JPG).

JPEG là một thuật toán nén ảnh phổ biến được sử dụng trong nhiều ứng dụng
thực tế, ví dụ như:

1. Lưu trữ hình ảnh trên thiết bị điện tử và trang web: Khi lưu trữ hình ảnh,
sử dụng thuật toán JPEG để giảm kích thước tệp tin hình ảnh và tiết
kiệm không gian lưu trữ. Điều này làm cho việc tải và hiển thị hình ảnh
trên các trang web hoặc thiết bị di động trở nên nhanh chóng hơn và tiết
kiệm băng thông Internet.

2. Xử lý hình ảnh trong các ứng dụng đòi hỏi tốc độ cao: JPEG là một
thuật toán nhanh và hiệu quả, vì vậy nó được sử dụng trong các ứng
dụng yêu cầu xử lý hình ảnh nhanh như xử lý video, chụp ảnh liên tục
trên máy ảnh số, và các ứng dụng thời gian thực khác.

3. Nén hình ảnh y tế: Các bức ảnh y tế như X-quang hoặc MRI có kích
thước lớn và phức tạp, vì vậy việc sử dụng thuật toán JPEG để nén các
ảnh này là rất quan trọng để giảm thiểu không gian lưu trữ và truyền tải
dữ liệu một cách nhanh chóng. Tuy nhiên, cần chú ý đến việc giữ lại độ
chính xác của hình ảnh y tế và tránh làm mất mát thông tin quan trọng
trong quá trình nén.

Các bước chính của thuật toán JPEG như sau:
1. Chia ảnh thành các khối 8x8 pixel.
2. Áp dụng biến đổi cosine rời rạc (DCT) lên từng khối để chuyển đổi ảnh
từ không gian thời gian sang không gian tần số.
3. Sử dụng các bộ lọc để loại bỏ thông tin không quan trọng trong không
gian tần số. Sử dụng mã hóa entropy để nén dữ liệu.
4. Mã hóa entropy sử dụng mã hóa Huffman để ánh xạ các ký tự vào các
mã độ dài khác nhau, giúp tối ưu hóa việc lưu trữ dữ liệu.
5. Lưu trữ các khối 8x8 pixel đã được nén theo thứ tự.

## 2. Ứng dụng này có cần phải tăng tốc không ?

Việc tăng tốc độ xử lý thuật toán JPEG là cần thiết khi chúng ta cần xây dựng
các ứng dụng yêu cầu xử lý hình ảnh nhanh như xử lý video, chụp ảnh liên tục trên
máy ảnh số và các ứng dụng thời gian thực khác đều yêu cầu thuật toán JPEG hoạt
động nhanh chóng để có thể cung cấp hình ảnh nhanh chóng và mượt mà.

## 3. Khó khăn mà nhóm có thể gặp phải

Thuật toán JPEG là một thuật toán tuần tự, có nghĩa là một số phần của nó
không thể được tăng tốc độ bằng cách sử dụng GPU. Vì vậy, chỉ một phần của thuật
toán có thể được tối ưu hóa để chạy trên GPU. Bên cạnh đó nhóm vẫn chưa xác định
được JPEG có phải là thuật toán nén có tốc độ xử lý tốt nhất trên GPU nên đây có thể
chưa phải là giải pháp tốt nhất.


## 4. Tài nguyên

Đồ án sẽ được viết bằng ngôn ngữ Python và sẽ thực thi trên nền tảng Google
Colab. Nhóm sẽ tìm hiểu và xây dụng về thuật toán nén JPEG từ các tài nguyên mạng
(viblo, github, ...) sau đó sẽ từ từ xây dựng phiên bản song song hóa và tối ưu dần
dần.

Một vài tài liệu mà nhóm đã tìm được:
http://pi.math.cornell.edu/~web6140/TopTenAlgorithms/JPEG.html?fbclid=IwAR2DR9plsb
Wu7Vd9enIbeycVg2L6I5K5PcuBI0yX1Qlg3Av3RWxKFvYtMA4
https://www.eecg.toronto.edu/~moshovos/CUDA08/arx/JPEG_report.pdf

## 5. Mục tiêu

### Plan to archive:
Xây dựng và hoàn thành thuật toán nén JPEG có thể đưa input vào ảnh sẽ xuất
ra ảnh chất lượng thấp hơn nhưng vẫn giữ nguyên những phần quan trọng của ảnh,
quan trọng là dung lượng ảnh giảm đi rất nhiều so với input và sản phẩm cần đạt hiệu
năng về tốc độ để có thể đáp ứng các ứng dụng, dự định nâng cao tốc độ xử lý theo
phương pháp thuật toán song song (Dùng GPU) nhanh hơn tối thiểu 2 lần so với thuật
toán tuần tự.

### Hope to archive:
Làm phiên bản thứ hai đạt kết quả tốt hơn phiên bản thứ nhất nhưng vẫn cho
kết quả đầu ra đúng với tiêu chuẩn về chất lượng ảnh cũng như dung lượng ảnh sau
khi nén và có thể có app giao diện để dễ dàng chạy chương trình


Trong trường hợp công việc chậm hơn ("75%"): Nhóm mong muốn có thể triển
khai phiên bản tuần tự của thuật toán nén ảnh JPEG, song song hóa một vài bước có
thể có trong thuật toán, đầu ra là ảnh có chất lượng không kém hơn nhiều so với ảnh
gốc và kích thước ảnh giảm.

[Link colab](https://colab.research.google.com/drive/1hASbTgy0KDWVjUZCzC_-opx6PiAa4Zg3?usp=sharing)
[Link Application](https://nguyenkhanh.pythonanywhere.com/)

