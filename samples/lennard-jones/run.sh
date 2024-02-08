python main.py --use_cuda
mv net.log output/gpu_loss_1000.log
mv time.log output/gpu_time_1000.log

python main.py
mv net.log output/сpu_loss_1000.log
mv time.log output/сpu_time_1000.log