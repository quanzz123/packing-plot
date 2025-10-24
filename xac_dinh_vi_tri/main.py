import cv2 as cv
import pickle as p

w , h = 107, 48

try:
    with open('viTri', 'rb') as f:
        VT = p.load(f)
except:
    VT = []

def mouseClick(event, x, y, flags, prams):
    if event == cv.EVENT_LBUTTONDOWN:
        VT.append((x,y))
    if event == cv.EVENT_RBUTTONDOWN:
        for i , vt in enumerate(VT):
            x1, y1 = vt
            if x1 < x < x1 + w and y1 < y < y1 + h:
                VT.pop(i)
    with open('viTri','wb') as f:
        p.dump(VT, f)
while True:
    img = cv.imread('Z:\\WORKSPACE\\bai_do_xe\\xac_dinh_vi_tri\\carParkImg.png')
    for vt in VT:
        cv.rectangle(img, vt, (vt[0]+w, vt[1]+h), (255,0,255), 2)
    cv.imshow('image', img)
    cv.setMouseCallback('image', mouseClick)
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()