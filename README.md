Giải pháp: chạy test_gstreamer dưới gdb (đã tạo + test xong)
Tôi đã tạo wrapper run_test_gstreamer_gdb.sh. Khi binary segfault, gdb tự in full backtrace vào 2 nơi:

pm2 error log (pm2 logs test_gstreamer --err)
file riêng có timestamp: build/crashlogs/crash_<thời gian>.log
rồi thoát → pm2 restart như cũ. Lúc chạy bình thường gdb không can thiệp (SIGPIPE/SIGUSR... được pass thẳng).

Áp dụng (chuyển pm2 sang wrapper)

pm2 delete test_gstreamer
pm2 start /home/orangepi/Documents/test_gstremer/run_test_gstreamer_gdb.sh --name test_gstreamer
pm2 save
Lần crash tiếp theo, xem:


pm2 logs test_gstreamer --err --lines 100      # backtrace hiện ngay đây
ls -t /home/orangepi/Documents/test_gstremer/build/crashlogs/   # hoặc file crash mới nhất
Backtrace sẽ cho dạng #0 SomeClass::method(...) at src/ai/Xxx.cpp:123 — chỉ thẳng dòng code C++ gây lỗi.

Hoàn tác (sau khi đã bắt được lỗi)

pm2 delete test_gstreamer
pm2 start /home/orangepi/Documents/test_gstremer/build/test_gstreamer --name test_gstreamer
pm2 save
Lưu ý
gdb thêm chút overhead ptrace (binary này đang 100% CPU), nên dùng để bắt crash tạm thời rồi revert.
Cách thay thế zero-overhead (core dump): không chạy dưới gdb, nhưng pm2 daemon đang có ulimit -c 0 nên phải nâng limit rồi restart cả pm2 daemon — phức tạp hơn. Khi đã có core thì phân tích bằng gdb build/test_gstreamer build/crashlogs/core.* -ex bt -ex quit. Wrapper gdb gọn hơn nên tôi khuyên dùng nó trước.
Bạn muốn tôi chạy luôn 3 lệnh áp dụng wrapper không? (sẽ restart test_gstreamer một lần). Hay bạn tự chạy để chủ động thời điểm?