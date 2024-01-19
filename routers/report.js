const express = require('express');
const path = require('path');
const router = express.Router();
const dfd = require('danfojs-node');

router.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/../html/report.html'));
});

router.get('/GetData', async (req, res) => {
  let file = 'predicted_size_distribution'; // 결과 파일명 넣기 *** 지금은 임시 test 파일
  let result;
  const filePath = path.join(__dirname, '/../storage/output/' + file + '.csv');
  await dfd.readCSV(filePath).then((df) => {
    result = {
      name: file,
      head: df.$columns,
      data: df.$data,
    };
  });
  res.status(200).send({ msg: 'test', tableData: result });
});

router.post('/GetColList', async (req, res) => {
  const { userID, projName } = req.body;
  const filePath = path.join(
    __dirname,
    '/../storage/output/report/' + projName + '.csv'
  );
  await dfd.readCSV(filePath).then((df) => {
    result = {
      head: df.$columns,
    };
  });
  res.status(200).json(result);
});

router.post('/GetDrawData', async (req, res) => {
  const { userID, projName, columnList, colCNT } = req.body;
  const filePath = path.join(
    __dirname,
    '/../storage/output/report/' + projName + '.csv'
  );
  await dfd.readCSV(filePath).then((df) => {
    let deletedCol = df.$columns;
    if (colCNT == 1) {
      deletedCol = deletedCol.filter((target) => target !== columnList);
    } else {
      for (let i = 0; i < columnList.length; i++) {
        deletedCol = deletedCol.filter((target) => target !== columnList[i]);
      }
    }
    df.drop({ columns: deletedCol, inplace: true });

    result = {
      head: df.$columns,
      data: df.$data,
    };
  });
  res.status(200).json(result);
});

module.exports = router;
