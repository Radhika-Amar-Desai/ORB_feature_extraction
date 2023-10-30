import cv2

static_image = cv2.imread("static_image.jpg")
static_image_gray = cv2.cvtColor(static_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures = 500)  

kp1, des1 = orb.detectAndCompute(static_image_gray, None)

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture(0)  
cap.set(3, 288)  
cap.set(4, 352) 

while True:
    ret, live_frame = cap.read()
    if not ret:
        break

    live_frame_gray = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(live_frame_gray, None)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)

        matched_image = cv2.drawMatches(static_image, kp1, live_frame, kp2, good_matches, None)

        cv2.imshow("Matches", matched_image)

    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()