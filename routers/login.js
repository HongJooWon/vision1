const express = require('express');
const path = require('path');
const router = express.Router();
const jwt = require('jsonwebtoken');
const chalk = require('chalk');
require('dotenv').config();

router.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '/../html/login.html'));
});

router.post('/Signin', (req, res) => {
  const { userID, password } = req.body;
  if (userID != 'admin') {
    console.log(chalk.red(`[LOGIN] ${userID} : Discorrect ID`));
    res.status(200).send({ Status: '001', Msg: 'ID 불일치' });
  } else if (password != 'Attic2018!@') {
    console.log(chalk.red(`[LOGIN] ${userID} : Discorrect Password`));
    res.status(200).send({ Status: '002', Msg: '패스워드 불일치' });
  } else {
    // const key = process.env.SECRET_KEY;
    // let token = jwt.sign(
    //   {
    //     type: 'JWT',
    //     userID: userID,
    //     userName: '관리자',
    //   },
    //   key,
    //   {
    //     expiresIn: '30m', // 만료시간 30분
    //     issuer: 'attic_admin',
    //   }
    // );
    console.log(chalk.green(`[LOGIN] ${userID} : Login Success`));
    res.status(200).send({
      Status: '000',
      userID: userID,
      userName: '관리자',
      // token: token,
    });
  }
});

module.exports = router;
