from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file
import os
import cv2
import numpy as np
import time
from ultralytics import YOLO
import math
import shutil
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

# تكوين المجلدات - تعديل المسارات للتأكد من أنها صحيحة
UPLOAD_FOLDER = os.path.abspath('uploads')
STATIC_FOLDER = os.path.abspath('static')
PROCESSED_VIDEOS_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos')
PLOTS_FOLDER = os.path.join(STATIC_FOLDER, 'plots')

app = Flask(__name__, static_folder=STATIC_FOLDER, static_url_path='/static')
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_VIDEOS_FOLDER'] = PROCESSED_VIDEOS_FOLDER
app.config['PLOTS_FOLDER'] = PLOTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # حد أقصى 64 ميجابايت للملفات المرفوعة

# التأكد من وجود المجلدات
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

# طباعة المسارات للتحقق منها
print(f"📁 مجلد الرفع: {UPLOAD_FOLDER}")
print(f"📁 مجلد الفيديوهات المعالجة: {PROCESSED_VIDEOS_FOLDER}")
print(f"📁 مجلد الرسوم البيانية: {PLOTS_FOLDER}")

# تحميل النموذج
model_path = "yolov8n-pose.pt"
try:
    model = YOLO(model_path)
    print(f"✅ تم تحميل النموذج من: {model_path}")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")
    model = None


def calculate_angle(a, b, c):
    """حساب الزاوية بين ثلاث نقاط"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def assess_squat(right_knee_angles, left_knee_angles):
    """
    تقييم حركة القرفصاء (Squat) بناءً على زوايا الركبتين

    معايير التقييم:
    1. الزاوية المثالية للركبة خلال القرفصاء: تتراوح بين 70 و 100 درجة
    2. تناسق الزوايا بين الركبتين: الفارق يجب أن يكون أقل من 15 درجة
    3. انخفاض صحيح: يجب أن تنخفض الزاوية لأقل من 110 درجة
    4. عودة للوضع العمودي: يجب أن تعود الزاوية لأكثر من 150 درجة
    """

    # تنظيف البيانات من القيم الفارغة
    valid_right = [angle for angle in right_knee_angles if angle is not None]
    valid_left = [angle for angle in left_knee_angles if angle is not None]

    if not valid_right or not valid_left:
        return {
            "status": "unknown",
            "message": "Squats could not be evaluated due to insufficient data for the knees.",
            "details": []
        }

    # البحث عن الزوايا الدنيا (أثناء الانخفاض)
    min_right = min(valid_right) if valid_right else 180
    min_left = min(valid_left) if valid_left else 180

    # البحث عن الزوايا القصوى (عند الوقوف)
    max_right = max(valid_right) if valid_right else 0
    max_left = max(valid_left) if valid_left else 0

    # الوسيط للزوايا الركبتين للمقارنة
    median_right = np.median(valid_right) if valid_right else 0
    median_left = np.median(valid_left) if valid_left else 0

    # تقييم النتائج
    results = []
    issues = []

    # تقييم الانخفاض المناسب (هل وصل للعمق المناسب)
    squat_depth_right = min_right < 110
    squat_depth_left = min_left < 110

    if not (squat_depth_right and squat_depth_left):
        issues.append("Not low enough")
        results.append("Dip: Not enough - knees not at the proper angle in the squat")
    else:
        results.append("Dip: Good - Proper depth reached in the squat")

    # تقييم العودة للوضع المستقيم
    stand_up_right = max_right > 150
    stand_up_left = max_left > 150

    if not (stand_up_right and stand_up_left):
        issues.append("Failure to return to an upright position")
        results.append("Standing: Incomplete - knees not fully straight")
    else:
        results.append("Standing: Good - Correctly returned to upright position")

    # تقييم التناسق بين الركبتين
    symmetry_diff = abs(median_right - median_left)
    if symmetry_diff > 15:
        issues.append("asymmetry between the knees")
        results.append(f"Consistency: Poor - Difference {symmetry_diff:.1f} degree between the knees")
    else:
        results.append("Coordination: Good - movement is coordinated between the knees")

    # تقييم زاوية الركبة (هل هي ضمن النطاق المثالي)
    if min_right < 70 or min_left < 70:
        issues.append("excessive knee drop")
        results.append("Knee angle: excessive drop - the knee is too low")
    elif min_right > 100 and min_left > 100:
        issues.append("Insufficient knee drop")
        results.append("Knee angle: Insufficient drop - The knee angle has not reached the ideal range.")
    else:
        results.append("Knee angle: Good - within the ideal range")

    # التقييم النهائي
    if len(issues) == 0:
        status = "correct"
        message = "Squats performed correctly"
    elif len(issues) <= 1:
        status = "partially_correct"
        message = "The squat exercise was partially performed, with some points for improvement."
    else:
        status = "incorrect"
        message = "There are several points that need improvement in performing the squat exercise."

    # إضافة المزيد من التفاصيل التقنية
    results.append(f"lowest right knee angle: {min_right:.1f} degree")
    results.append(f"lowest angle of the left knee: {min_left:.1f} degree")

    return {
        "status": status,
        "message": message,
        "details": results
    }


def process_video(input_path):
    """معالجة الفيديو وحساب زوايا الركبتين فقط وإنشاء رسم بياني لها"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("فشل في فتح الفيديو")

        # الحصول على معلومات الفيديو
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # التأكد من أن معدل الإطارات منطقي
        if fps <= 0:
            fps = 30  # قيمة افتراضية معقولة

        # إنشاء اسم الملف الجديد
        timestamp = int(time.time())
        output_filename = f'processed_{timestamp}.mp4'
        output_path = os.path.join(app.config['PROCESSED_VIDEOS_FOLDER'], output_filename)

        # تنظيف أي ملفات سابقة (إذا لزم الأمر) لتجنب تضارب الأسماء
        if os.path.exists(output_path):
            os.remove(output_path)

        # طباعة مسار الملف الناتج للتحقق منه
        print(f"📹 مسار الفيديو الناتج: {output_path}")

        # إنشاء كائن VideoWriter باستخدام ترميز أكثر توافقية (MPEG-4)
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # استخدام H.264 بدلا من mp4v
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            # محاولة استخدام ترميز بديل
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("فشل في إنشاء ملف الفيديو الناتج")

        frame_count = 0
        max_frames = 1000  # حد أقصى للإطارات لمنع الفيديوهات الطويلة جدا

        # متغيرات لتتبع زوايا الركبة على مدار الفيديو
        right_knee_angles = []
        left_knee_angles = []
        frame_numbers = []

        # معالجة كل إطار
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # تطبيق نموذج YOLOv8 للكشف عن وضعية الجسم
            results = model(frame, conf=0.3)

            # معالجة النتائج
            for result in results:
                if result.keypoints is not None and result.keypoints.shape[1] > 0:
                    keypoints = result.keypoints.data[0]  # أخذ أول شخص تم رصده

                    # التحقق من أن النقاط الرئيسية موجودة
                    if len(keypoints) >= 17:  # COCO format has 17 keypoints
                        # حساب زاوية الركبة اليمنى فقط
                        right_knee_angle = None
                        if all(keypoints[[12, 14, 16], 2] > 0.5):
                            right_knee_angle = calculate_angle(
                                keypoints[12][:2].tolist(),
                                keypoints[14][:2].tolist(),
                                keypoints[16][:2].tolist()
                            )
                            cv2.putText(frame, f"{int(right_knee_angle)}",
                                        tuple(map(int, keypoints[14][:2])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            right_knee_angles.append(right_knee_angle)
                        else:
                            right_knee_angles.append(None)

                        # حساب زاوية الركبة اليسرى فقط
                        left_knee_angle = None
                        if all(keypoints[[11, 13, 15], 2] > 0.5):
                            left_knee_angle = calculate_angle(
                                keypoints[11][:2].tolist(),
                                keypoints[13][:2].tolist(),
                                keypoints[15][:2].tolist()
                            )
                            cv2.putText(frame, f"{int(left_knee_angle)}",
                                        tuple(map(int, keypoints[13][:2])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            left_knee_angles.append(left_knee_angle)
                        else:
                            left_knee_angles.append(None)

                # رسم الهيكل العظمي
                annotated_frame = result.plot()
                out.write(annotated_frame)
                frame_numbers.append(frame_count)
                frame_count += 1

        # إغلاق الملفات والتنظيف
        cap.release()
        out.release()

        # تقييم القرفصاء
        squat_assessment = assess_squat(right_knee_angles, left_knee_angles)

        # إنشاء رسم بياني للزوايا
        plot_filename = f'knee_angles_plot_{timestamp}.png'
        plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)

        plt.figure(figsize=(10, 6))
        plt.title('Knee angles through the analysis', fontsize=16)
        plt.xlabel('frames', fontsize=12)
        plt.ylabel('Angles (in degrees)', fontsize=12)

        # تنظيف البيانات وإزالة القيم الفارغة
        valid_frames_right = [(i, angle) for i, angle in zip(frame_numbers, right_knee_angles) if angle is not None]
        valid_frames_left = [(i, angle) for i, angle in zip(frame_numbers, left_knee_angles) if angle is not None]

        if valid_frames_right:
            frames_r, angles_r = zip(*valid_frames_right)
            plt.plot(frames_r, angles_r, 'r-', label='Right knee angle')

        if valid_frames_left:
            frames_l, angles_l = zip(*valid_frames_left)
            plt.plot(frames_l, angles_l, 'b-', label='Left knee angle')

        # إضافة منطقة للزاوية المثالية للقرفصاء
        plt.axhspan(70, 100, alpha=0.2, color='green', label='Ideal squat angle')

        plt.grid(True)
        plt.legend(loc='best')
        plt.tight_layout()

        # ضبط اتجاه الرسم البياني ليكون من اليمين إلى اليسار
        plt.gca().invert_xaxis()

        # حفظ الرسم البياني
        plt.savefig(plot_path)
        plt.close()

        # محاولة تحويل الفيديو إلى تنسيق أكثر توافقية باستخدام FFmpeg إذا كان متاحاً
        try:
            import subprocess
            compat_filename = f'web_compatible_{timestamp}.mp4'
            compat_path = os.path.join(app.config['PROCESSED_VIDEOS_FOLDER'], compat_filename)

            # محاولة تنفيذ FFmpeg لإعادة ترميز الفيديو ليكون متوافقاً مع المتصفحات
            ffmpeg_cmd = [
                'ffmpeg', '-i', output_path,
                '-vcodec', 'libx264', '-acodec', 'aac',
                '-strict', 'experimental', '-pix_fmt', 'yuv420p',
                compat_path
            ]

            subprocess.run(ffmpeg_cmd, check=True)

            # إذا نجح التحويل، استخدم الملف الجديد
            if os.path.exists(compat_path) and os.path.getsize(compat_path) > 0:
                print(f"✅ تم تحويل الفيديو بنجاح إلى تنسيق متوافق: {compat_filename}")
                os.remove(output_path)  # حذف الملف القديم
                output_filename = compat_filename
            else:
                print("⚠️ فشل تحويل الفيديو، سيتم استخدام الملف الأصلي")
        except Exception as e:
            print(f"⚠️ لم يتم تحويل الفيديو: {e}")
            print("⚠️ سيتم استخدام الملف الأصلي")

        # التحقق من وجود الملف بعد المعالجة
        final_output_path = os.path.join(app.config['PROCESSED_VIDEOS_FOLDER'], output_filename)

        print(f"✔️ تم معالجة الفيديو بنجاح: {output_filename}")

        if os.path.exists(final_output_path):
            print(f"✅ ملف الفيديو موجود في: {final_output_path}")
            file_size = os.path.getsize(final_output_path)
            print(f"📊 حجم الملف: {file_size} بايت")
        else:
            print(f"❌ ملف الفيديو غير موجود في: {final_output_path}")

        # نرجع اسم الملف الذي تم إنشاؤه والرسم البياني وتقييم القرفصاء
        return output_filename, plot_filename, squat_assessment

    except Exception as e:
        print(f"❌ خطأ أثناء معالجة الفيديو: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video_file = request.files.get('video')

        if not video_file:
            flash('لم يتم اختيار ملف فيديو.', 'error')
            return redirect(request.url)

        # حفظ الفيديو في المجلد المخصص
        filename = f"input_{int(time.time())}_{video_file.filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(file_path)

        # معالجة الفيديو
        processed_filename, plot_filename, squat_assessment = process_video(file_path)

        if not processed_filename:
            flash('حدث خطأ أثناء معالجة الفيديو، يرجى المحاولة لاحقاً.', 'error')
            return redirect(request.url)

        # التحقق من وجود الملف المعالج قبل إرجاع القالب
        processed_file_path = os.path.join(app.config['PROCESSED_VIDEOS_FOLDER'], processed_filename)
        if os.path.exists(processed_file_path):
            print(f"✅ الملف المعالج موجود: {processed_file_path}")
        else:
            print(f"❌ الملف المعالج غير موجود: {processed_file_path}")
            flash('تم معالجة الفيديو لكن لم يتم العثور على الملف الناتج.', 'error')
            return redirect(request.url)

        # تمرير اسم الملف إلى القالب
        video_url = url_for('static', filename=f'processed_videos/{processed_filename}')
        plot_url = url_for('static', filename=f'plots/{plot_filename}') if plot_filename else None

        return render_template('index.html',
                               video_path=video_url,
                               video_filename=processed_filename,
                               plot_path=plot_url,
                               squat_assessment=squat_assessment,
                               absolute_path=processed_file_path)

    return render_template('index.html')


@app.route('/download/<filename>')
def download_video(filename):
    """تنزيل الفيديو المعالج"""
    return send_from_directory(app.config['PROCESSED_VIDEOS_FOLDER'],
                               filename,
                               mimetype='video/mp4',
                               as_attachment=True)


@app.route('/download_plot/<filename>')
def download_plot(filename):
    """تنزيل الرسم البياني"""
    return send_from_directory(app.config['PLOTS_FOLDER'],
                               filename,
                               mimetype='image/png',
                               as_attachment=True)


@app.route('/view/<filename>')
def view_video(filename):
    """عرض الفيديو المعالج مباشرة"""
    return send_from_directory(app.config['PROCESSED_VIDEOS_FOLDER'],
                               filename,
                               mimetype='video/mp4')


if __name__ == '__main__':
    app.run(debug=True)