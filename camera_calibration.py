import numpy as np
import cv2

def run_calibration():
    # === 設定參數 ===
    # 棋盤格的「內角點」數量 (寬, 高) -> 注意：是方格交界點的數量，不是方格數
    # 如果您的棋盤格是 10x7 個格子，那內角點就是 9x6
    CHECKERBOARD = (9, 6) 
    
    # 每個方格的邊長 (單位：mm 或 cm 都可以，這會影響 tvec 的單位，但不影響內參 K)
    SQUARE_SIZE = 25.0 

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 準備世界座標系中的點 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # 儲存偵測到的點
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D points in image plane.

    cap = cv2.VideoCapture(0)
    # 設定解析度 (需與您跑 SLAM 的解析度一致)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"=== 相機校正工具 ===")
    print(f"1. 請將 {CHECKERBOARD} 棋盤格置於鏡頭前")
    print("2. 按下 'c' 鍵拍攝一張 (至少需要 15 張不同角度)")
    print("3. 按下 'q' 鍵結束拍攝並開始計算")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp_frame = frame.copy()

        # 尋找棋盤格角點
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, 
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                         cv2.CALIB_CB_FAST_CHECK + 
                                                         cv2.CALIB_CB_NORMALIZE_IMAGE)

        # 如果找到，畫出來給使用者看
        if ret_corners:
            cv2.drawChessboardCorners(disp_frame, CHECKERBOARD, corners, ret_corners)
            cv2.putText(disp_frame, "Ready to Capture!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(disp_frame, f"Captured: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Calibration', disp_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ret_corners:
            # 增加精確度 (Sub-pixel accuracy)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            count += 1
            print(f"已拍攝第 {count} 張")
            # 閃爍一下畫面提示
            cv2.imshow('Calibration', 255 - frame)
            cv2.waitKey(50)
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < 5:
        print("拍攝張數太少，無法校正 (至少需要 5 張以上)。")
        return

    print("\n正在計算相機矩陣... 請稍候...")
    
    # === 核心校正函式 ===
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n" + "="*40)
    print("校正結果 (Calibration Results)")
    print("="*40)
    print(f"投影誤差 (Reprojection Error): {ret:.4f} (越小越好，< 1.0 為佳)")
    
    print("\n[內參矩陣 K] (請複製這個到 SLAM 程式):")
    print("-" * 20)
    print(f"Focal Length (fx, fy): {mtx[0,0]:.4f}, {mtx[1,1]:.4f}")
    print(f"Principal Point (cx, cy): {mtx[0,2]:.4f}, {mtx[1,2]:.4f}")
    print("-" * 20)
    print("完整矩陣:")
    print(mtx)

    print("\n[畸變係數 Distortion Coefficients] (k1, k2, p1, p2, k3):")
    print(dist)

    # 簡單計算視角 (FOV)
    h, w = gray.shape
    fx = mtx[0,0]
    fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
    print(f"\n估計水平視角 (FOV X): {fov_x:.2f} 度")

if __name__ == '__main__':
    run_calibration()