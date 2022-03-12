# Randyxiao 2021-6-10

# Normal message
# {
#   'type': int,
#   'msg' : ...,
#   'tag' : str  
# }

# Connection
IP="192.168.43.2"



MSG_CONFIG = 1
MSG_TEST_COMM_SPEED = 2
MSG_DEPLOY_DATA=3
MSG_END_EXPERIMENT=4
MSG_LOCAL_TRAIN=5
MSG_GET_GRAD=6
MSG_TRAIN_RESULT=7
MSG_EXP_START=8
MSG_EXP_END=9

SERVER = 'server'
CLIENT = 'client'
SERVER_SENDER = {'type':SERVER, 'num':0}

TAG_SIZE=50000
COMM_TEST_EXP=50
# file buffer

BYTE = 1
HEAD_SIZE = 8
FILE_BUF_SIZE = 1024

# Signal 
FIRST_SHAKE_HANDS='a'
SECOND_SHAKE_HANDS='b'
