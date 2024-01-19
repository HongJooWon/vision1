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
        '/../storage/input/' + req.body.projName + '/'
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
  res.sendFile(path.join(__dirname, '/../html/speed.html'));
});
router.get('/result', (req, res) => {
  res.sendFile(path.join(__dirname, '/../html/speedresult.html'));
});
router.get('/report', (req, res) => {
  res.sendFile(path.join(__dirname, '/../html/speedreport.html'));
});

router.post('/GetBatchList', async (req, res) => {
  const { userID } = req.body;

  const folder = './storage/input';
  fs.readdir(folder, (err, filelist) => {
    // console.log(filelist);
    res.status(200).send({ fileList: filelist });
  });
});

router.post('/GetImageList', async (req, res) => {
  const { userID, folderName } = req.body;
  const folder = './storage/input/' + folderName;
  fs.readdir(folder, (err, filelist) => {
    res.status(200).send({ fileList: filelist });
  });
});

router.post('/GetResultImage', async (req, res) => {
  const { userID, folderName } = req.body;
  const folder = './storage/output/sentimentation/images/' + folderName;
  fs.readdir(folder, (err, filelist) => {
    res.status(200).send({ fileList: filelist });
  });
});

router.post('/uploadFile', upload.array('file'), (req, res) => {
  const { userID, projName } = req.body;
  const addrList = [
    path.join(
      __dirname,
      '/../storage/output/sentimentation/csv/' + projName + '/'
    ),
    path.join(
      __dirname,
      '/../storage/output/sentimentation/images/' + projName + '/'
    ),
  ];
  for (let i = 0; i < addrList.length; i++) {
    if (!fs.existsSync(addrList[i])) {
      fs.mkdirSync(addrList[i], { recursive: true });
    }
  }
  const basePath = path.join(__dirname, '/../storage');
  const python = spawn('python', [
    path.join(__dirname, '/../storage/python/vision_solution_v8_1.py'),
    basePath,
    projName,
  ]);
  python.stdout.on('data', async (data) => {
    console.log(data.toString());
    if (data.toString() == 'The end') {
      fs.readdir(folderPath, (err, filelist) => {
        res.send({ fileList: filelist });
      });
    }
  });
});

router.post('/GetSpeed', async (req, res) => {
  const { userId, batchName } = req.body;

  const filePath = path.join(
    __dirname,
    '/../storage/output/sentimentation/speed.csv'
  );
  let value = 0;
  await dfd.readCSV(filePath).then((df) => {
    for (let i = 0; i < df.$data.length; i++) {
      if (df.$data[i][0] == batchName) {
        value = df.$data[i][1];
        break;
      }
    }
  });

  res.send({ speed: value });
});

router.post('/GetDrawData', async (req, res) => {
  const { userId, batchName } = req.body;

  let filePath = path.join(
    __dirname,
    '/../storage/output/sentimentation/speed.csv'
  );
  let value = 0;
  let result;
  await dfd.readCSV(filePath).then((df) => {
    for (let i = 0; i < df.$data.length; i++) {
      if (df.$data[i][0] == batchName) {
        value = df.$data[i][1];
        break;
      }
    }
  });
  filePath = path.join(
    __dirname,
    '/../storage/output/sentimentation/csv/' +
      batchName +
      '/floc_height_' +
      batchName +
      '.csv'
  );
  await dfd.readCSV(filePath).then((df) => {
    result = {
      head: df.$columns,
      data: df.$data,
      speed: value,
    };
  });
  res.json(result);
});

module.exports = router;
