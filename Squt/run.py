from app import app

if __name__ == '__main__':
    # تشغيل التطبيق على المنفذ 5000 مع تفعيل وضع تصحيح الأخطاء وإعادة التحميل التلقائي
    app.run(host='0.0.0.0', port=5000, debug=True)