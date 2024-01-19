const express = require('express');
const path = require('path');
const router = express.Router();
const fs = require('fs');
const multer = require('multer');
const upload = multer({
  storage: multer.diskStorage({
    destination: function (req, file, done) {
      done(null, 'data/images/img/');
    },
    filename: function (req, file, done) {
      const ext = path.extname(file.originalname);
      done(null, path.basename(file.originalname, ext) + Date.now() + ext);
    },
  }),
});

router.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/../html/viewer.html'));
});

router.post('/upload', upload.single('image'), (req, res) => {
  console.log(req.file, req.body);
  res.json({ msg: 'success', path: req.file.path });
});

router.get('/getFilelist', (req, res) => {
  const folder = './data/images/img';
  fs.readdir(folder, (err, filelist) => {
    res.json(filelist);
  });
});

module.exports = router;
