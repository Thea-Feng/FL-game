#!/bin/bash

USERNAME="pi"
HOSTS="192.168.43.21"
SCRIPT="pwd; ls"

# copy ssh-key
for HOSTNAME in ${HOSTS} ; do
    ssh-copy-id -i ~/.ssh/id_rsa.pub pi@${HOSTNAME}
    # ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/fed-iot-sci"
done

# remove old version
for HOSTNAME in ${HOSTS} ; do
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/pi/fed-iot/fed-iot-sci; mkdir /home/pi/fed-iot"
    echo ${HOSTNAME}
done

# Transfer new version
# cd ..
DIR="$( cd "$( dirname "$0" )" && pwd )"
echo ${DIR}

for HOSTNAME in ${HOSTS} ; do
    scp -rp "${DIR}" pi@${HOSTNAME}:/home/pi/fed-iot/
    # ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/fed-iot-sci"
done

# Install necessary library
for HOSTNAME in ${HOSTS} ; do
    ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "sudo su - <<'EOF' 
    python -m pip install pickle5
    python -m pip install tensorboardX"
    echo ${HOSTNAME}
done