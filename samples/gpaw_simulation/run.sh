python main.py ./gpaw/Cu111.txt -с
mv net.log output/сpu_loss_100.log
mv time.log output/сpu_time_100.log

python main.py ./gpaw/Cu111.txt
mv net.log output/сpu_loss_100.log
mv time.log output/сpu_time_100.log