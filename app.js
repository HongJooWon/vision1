const express = require('express');
const http = require('http');
const bodyParser = require('body-parser');
const app = express();
const path = require('path');
require('dotenv').config();

app.use(bodyParser.urlencoded({ extended: false }));

const server = http.createServer(app);

app.use(express.static(__dirname));
console.log(__dirname);

const loginRouter = require('./routers/login');
const mainRouter = require('./routers/main');
const viewerRouter = require('./routers/viewer');
const measureRouter = require('./routers/measurement');
const reportRouter = require('./routers/report');
const speedRouter = require('./routers/speed');
const settingRouter = require('./routers/setting');
server.listen(process.env.PORT, () =>
  console.log(`Listening Port ${process.env.PORT}`)
);

app.get('/', (req, res) => {
  res.sendFile(__dirname + '/html/login.html');
});

app.use('/login', loginRouter);
app.use('/main', mainRouter);
app.use('/viewer', viewerRouter);
app.use('/measurement', measureRouter);
app.use('/report', reportRouter);
app.use('/setting', settingRouter);
app.use('/speed', speedRouter);

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`)
})

module.exports = app;