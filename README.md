# Purpose: Capture Human Hand Shape and Mimic on the Dexhand 

## step 1: to detect the hand from the camera or picture

## step 2: to convert the hand marker into the parameter of robot hand

## Usage
- for the photo, you can run with the hand_detector_static.py, but currently only the 1st found hand can be mimic. 
- for the video, you can run with the hand_detector.py, but there might be some issue. (have no time to fix it. )
- the hand_visualn should be put in the pyzlg_dexhand folder, because we need the pyzlg_dexhand(link below) to handle the hand

# configure: 
1. hardware: dexhand 021 MP, and below is the code for that hand
https://gitee.com/dexrobot/pyzlg_dexhand

2. software: 
- linux: Ubuntu 22.04
- python 3.12
- need zlg driver to connect with dexhand. 
    - notice: after you install the zlg driver, you may want to change the priviledge of the usb dev.
        - below command need every time after you connect the hand to pc.
            - lsusb # this is to find what is the usb for the dexhand or zlg
            - sudo chmod 666 /dev/bus/usb/00x/00y # this is to change the priviledge of the dev and try to control it. 

        - or you can just change the dev file for permenent. 
            - change the mode to 0666 in /etc/udev/rules.d/99-zlg-can.rules

3. When I tunning it, I put the hand_visual under the pyzlg_dexhand/tools. Just for reference

4. demo video with this hand for reference: (hope you can see it, cross fingers)
https://www.bilibili.com/video/BV1Sv6vB2EzW/

# to be optimized (later if there is any chance)

- the hand shape convertion still have space to be optimized
- the number of the hand can be mimic either in a photo or in a video