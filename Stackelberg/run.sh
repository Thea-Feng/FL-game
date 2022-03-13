
#!/bin/bash

# run nano
USERNAME="nano"
HOSTS="192.168.43.10 192.168.43.11 192.168.43.12 192.168.43.13 192.168.43.14 192.168.43.15 192.168.43.16 192.168.43.17 192.168.43.18 192.168.43.19"
# HOSTS="192.168.43.10"
# HOSTS="192.168.43.10 192.168.43.11 192.168.43.12"
# execute program
for HOSTNAME in ${HOSTS} ; do
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "cd /home/nano/fed-iot/fed-iot-sci; python app.py --mode client > log.txt &"
    echo ${HOSTNAME}
    sleep 2
done

# run pi
USERNAME="pi"
HOSTS="192.168.43.20 192.168.43.21 192.168.43.22 192.168.43.23 192.168.43.24 192.168.43.25 192.168.43.26 192.168.43.27 192.168.43.28 192.168.43.29 192.168.43.30 192.168.43.31 192.168.43.32 192.168.43.33 192.168.43.34 192.168.43.35 192.168.43.36 192.168.43.37 192.168.43.38 192.168.43.39"

# HOSTS="192.168.43.33"

#  HOSTS="192.168.43.21 192.168.43.22 192.168.43.23 192.168.43.24 192.168.43.25 192.168.43.26 192.168.43.27 192.168.43.28 192.168.43.29 192.168.43.30 192.168.43.31 192.168.43.32 192.168.43.34 192.168.43.35 192.168.43.36 192.168.43.37 192.168.43.38 192.168.43.39"


# execute program
for HOSTNAME in ${HOSTS} ; do

    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "sudo su - <<'EOF' 
    cd /home/Stackelberg
    python main_bench.py --model server > log.txt &
    python main_property.py --model server > log.txt &"

    echo ${HOSTNAME}
    sleep 2
done