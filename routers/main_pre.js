const express = require('express');
const path = require('path');
const router = express.Router();
const fs = require('fs');
const multer = require('multer');
const dfd = require('danfojs-node');

const upload = multer({
  storage: multer.diskStorage({
    destination: function (req, file, done) {
      done(null, 'data/20230702/');
    },
    filename: function (req, file, done) {
      const ext = path.extname(file.originalname);
      done(null, path.basename(file.originalname, ext) + Date.now() + ext);
    },
  }),
});

router.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/../html/main.html'));
});

router.post('/upload', upload.array('image'), (req, res) => {
  res.status(200).send({ msg: 'success', fileInfo: req.files });
  // res.json({ msg: 'success', path: req.file.path });
});

router.post('/GetDetection', async (req, res) => {
  const { fileName } = req.body;

  // fileName에 따라서 콩나물 또는 미생물 결과 사진과 데이터 전달
  // 배열에 사진 경로 담아서 imgData로 보내기
  // 결과 데이터는 컬럼과 데이터로 구분해서 tableData로 보내기
  const folder = './data/bacteria_detected';
  let list = [];
  fs.readdir(folder, (err, filelist) => {
    filelist.forEach((el) => {
      list.push('/.' + folder + '/' + el);
    });
  });

  let file = 'test'; // 결과 파일명 넣기 *** 지금은 임시 test 파일
  let result;
  const filePath = path.join(__dirname, '/../data/result/' + file + '.csv');
  await dfd.readCSV(filePath).then((df) => {
    result = {
      name: file,
      head: df.$columns,
      data: df.$data,
    };
  });
  res.status(200).send({ msg: 'test', tableData: result, imgData: list });
});

module.exports = router;
