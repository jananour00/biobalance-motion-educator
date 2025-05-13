from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import time

# تكوين المجلدات
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
PROCESSED_VIDEOS_FOLDER = os.path.join(STATIC_FOLDER, 'processed_videos')

app = Flask(__name__, static_folder=STATIC_FOLDER)
app.secret_key = 'your_secret_key_here'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_VIDEOS_FOLDER'] = PROCESSED_VIDEOS_FOLDER

# التأكد من وجود المجلدات
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_VIDEOS_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # إنشاء ملف نصي للاختبار
    test_filename = f'test_{int(time.time())}.txt'
    test_path = os.path.join(app.config['PROCESSED_VIDEOS_FOLDER'], test_filename)
    
    # كتابة محتوى اختباري في الملف
    with open(test_path, 'w') as f:
        f.write("هذا ملف اختباري")
    
    # طباعة معلومات التشخيص
    print(f"المسارات:")
    print(f"1. مجلد الرفع: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"2. المجلد الثابت: {os.path.abspath(STATIC_FOLDER)}")
    print(f"3. مجلد الفيديوهات المعالجة: {os.path.abspath(PROCESSED_VIDEOS_FOLDER)}")
    print(f"4. ملف الاختبار: {os.path.abspath(test_path)}")
    print(f"5. هل ملف الاختبار موجود؟ {os.path.exists(test_path)}")
    
    # المسارات من وجهة نظر Flask
    static_url = url_for('static', filename=f'processed_videos/{test_filename}')
    direct_url = url_for('processed_video', filename=test_filename)
    
    print(f"6. رابط الملف الثابت: {static_url}")
    print(f"7. رابط الملف المباشر: {direct_url}")
    
    # قائمة الملفات في مجلد الفيديوهات المعالجة
    print("8. قائمة الملفات في مجلد الفيديوهات المعالجة:")
    for file in os.listdir(PROCESSED_VIDEOS_FOLDER):
        file_path = os.path.join(PROCESSED_VIDEOS_FOLDER, file)
        print(f"   - {file} ({os.path.getsize(file_path)} بايت)")
    
    return render_template('debug.html', 
                          test_filename=test_filename,
                          static_url=static_url,
                          direct_url=direct_url)

@app.route('/processed_videos/<filename>')
def processed_video(filename):
    print(f"محاولة الوصول إلى الملف: {filename}")
    print(f"المسار الكامل: {os.path.join(app.config['PROCESSED_VIDEOS_FOLDER'], filename)}")
    print(f"هل الملف موجود؟ {os.path.exists(os.path.join(app.config['PROCESSED_VIDEOS_FOLDER'], filename))}")
    return send_from_directory(app.config['PROCESSED_VIDEOS_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5001)