const express = require('express');
const path = require('path');
const router = express.Router();
const fs = require('fs');
const multer = require('multer');
const dfd = require('danfojs-node');
const spawn = require('child_process').spawn;

const upload = multer({
  storage: multer.diskStorage({
    destination: function (req, file, done) {
      const dir = path.join(
        __dirname,
        '/../storage/yolo_data/test/images/' + req.body.projName + '/'
      );
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      done(null, dir);
      // done(null, 'storage/yolo_data/test/images/' + req.body.projName + '/');
    },
    filename: function (req, file, done) {
      done(null, file.originalname);
    },
  }),
});

router.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/../html/main.html'));
});

// router.post('/upload', upload.array('image'), (req, res) => {
//   res.status(200).send({ msg: 'success', fileInfo: req.files });
//   // res.json({ msg: 'success', path: req.file.path });
// });

router.post('/GetDataList', async (req, res) => {
  const { userID } = req.body;

  const folder = './storage/yolo_data/test/images';
  fs.readdir(folder, (err, filelist) => {
    // console.log(filelist);
    res.status(200).send({ fileList: filelist });
  });
});

router.post('/GetImageList', async (req, res) => {
  const { userID, folderName } = req.body;
  const folder = './storage/yolo_data/test/images/' + folderName;
  fs.readdir(folder, (err, filelist) => {
    // console.log(filelist);
    res.status(200).send({ fileList: filelist });
  });
});

router.post('/uploadFile', upload.array('file'), (req, res) => {
  const { userID, projName } = req.body;
  const addrList = [
    path.join(
      __dirname,
      '/../storage/output/bbox_plotted/gt/' + projName + '/'
    ),
    path.join(
      __dirname,
      '/../storage/output/bbox_plotted/pred/' + projName + '/'
    ),
    path.join(__dirname, '/../storage/output/hist/gt/' + projName + '/'),
    path.join(__dirname, '/../storage/output/hist/pred/' + projName + '/'),
  ];
  for (let i = 0; i < addrList.length; i++) {
    if (!fs.existsSync(addrList[i])) {
      fs.mkdirSync(addrList[i], { recursive: true });
    }
  }

  const folderPath = path.join(
    __dirname,
    '/../storage/yolo_data/test/images/' + projName
  );

  const basePath = path.join(__dirname, '/../storage');
  const python = spawn('python', [
    path.join(__dirname, '/../storage/python/vision_solution_v8.py'),
    basePath,
    folderPath,
    projName,
  ]);
  python.stdout.on('data', async (data) => {
    console.log(data.toString());
    if (data.toString() == 'The End') {
      fs.readdir(folderPath, (err, filelist) => {
        // console.log(filelist);
        res.send({ fileList: filelist });
      });
    }
  });
});

module.exports = router;
