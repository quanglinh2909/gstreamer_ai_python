module.exports = {
  apps: [
    {
      name: 'gstreamer-pyhthon',
      script: './main.py',
      interpreter: './env/bin/python',  // dùng tương đối theo cwd
      cwd: './',  // hoặc bỏ qua nếu file ecosystem nằm trong cùng thư mục
      autorestart: true,
      watch: false,
    },
  ],
};
