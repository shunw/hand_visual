# Purpose: Capture Human Hand Shape and Mimic on the Dexhand 

## step 1: to detect the hand from the camera or picture

## step 2: to convert the hand marker into the parameter of robot hand

## for the photo, you can run with the hand_detector_static

# configure: 
1. hardware: dexhand, and below is the code for that hand
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
