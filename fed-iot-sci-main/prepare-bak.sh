
#!/bin/bash

# USERNAME="nano"
# HOSTS="192.168.43.10 192.168.43.11 192.168.43.12 192.168.43.13 192.168.43.14 192.168.43.15 192.168.43.16 192.168.43.17 192.168.43.18 192.168.43.19"
# # HOSTS="192.168.43.10"
# SCRIPT="pwd; ls"

# # Transfer new version
# # cd ..
# DIR="$( cd "$( dirname "$0" )" && pwd )"
# echo ${DIR}

# # copy ssh-key
# for HOSTNAME in ${HOSTS} ; do
#     echo ${HOSTNAME}
#     # Copy ssh passwd
#     # ssh-copy-id nano@${HOSTNAME}

#     # Remove Old version Code
#     ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/fed-iot-sci"
    
#     # Upload New version Code 
#     # scp -rp "${DIR}" nano@${HOSTNAME}:/home/nano/fed-iot

#     # Install Necessary Package
#     # ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "python -m pip install pickle5; python -m pip install tensorboardX"

# done

# # remove old version
# for HOSTNAME in ${HOSTS} ; do
#     ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/fed-iot-sci"
#     echo ${HOSTNAME}
# done


# for HOSTNAME in ${HOSTS} ; do
#     scp -rp "${DIR}" nano@${HOSTNAME}:/home/nano/fed-iot
#     # ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/fed-iot-sci"
# done

# # Install necessary library
# for HOSTNAME in ${HOSTS} ; do
#     ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "python -m pip install pickle5; python -m pip install tensorboardX"
#     echo ${HOSTNAME}
# done

# # prepare pi
# USERNAME="pi"
# HOSTS="192.168.43.21 192.168.43.22 192.168.43.23 192.168.43.24 192.168.43.25 192.168.43.26 192.168.43.27 192.168.43.28 192.168.43.29 192.168.43.30 192.168.43.31 192.168.43.32 192.168.43.33 192.168.43.34 192.168.43.35 192.168.43.36 192.168.43.37 192.168.43.38 192.168.43.39"
# SCRIPT="pwd; ls"

# # copy ssh-key
# for HOSTNAME in ${HOSTS} ; do
#     ssh-copy-id pi@${HOSTNAME}
#     # ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/fed-iot-sci"
# done

# # remove old version
# for HOSTNAME in ${HOSTS} ; do
#     ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/pi/fed-iot/fed-iot-sci; mkdir /home/pi/fed-iot"
#     echo ${HOSTNAME}
# done

# # Transfer new version
# # cd ..
# DIR="$( cd "$( dirname "$0" )" && pwd )"
# echo ${DIR}

# for HOSTNAME in ${HOSTS} ; do
#     scp -rp "${DIR}" pi@${HOSTNAME}:/home/pi/fed-iot/
#     # ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "rm -rf /home/nano/fed-iot/fed-iot-sci"
# done

# # Install necessary library
# for HOSTNAME in ${HOSTS} ; do
#     ssh -o StrictHostKeyChecking=no -l ${USERNAME} ${HOSTNAME} "sudo su - <<'EOF' 
#     python -m pip install pickle5
#     python -m pip install tensorboardX"
#     echo ${HOSTNAME}
# done