// 사용자 정보 없으면
if (!window.localStorage.getItem('user')) {
  location.href = '/login';
}

let user = JSON.parse(localStorage.getItem('user'));
let dataList = [];
let project = '';
$(document).ready(() => {
  $('#userName').text(user.userName);
  getDataList();

  $('#searchWord').on('keyup', (e) => {
    let word = $('#searchWord').val();
    console.log(word);

    if (word == '') {
      $('#viewer_datalist li').show();
      $('.searchDelete').hide();
    } else {
      $('.searchDelete').show();
      $('#viewer_datalist li').hide();
      $('#viewer_datalist li:contains(' + word + ')').show();
    }
  });

  $('.searchDelete').on('click', () => {
    $('#searchWord').val('');
    $('#viewer_datalist li').show();
    $('.searchDelete').hide();
  });
});

const getDataList = () => {
  const param = {
    userID: user.userID,
  };

  $.ajax({
    type: 'post',
    url: '/main/GetDataList',
    dataType: 'json',
    data: param,
    success: (res) => {
      if (res) {
        setDataList(res.fileList);
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};

const setDataList = (_res) => {
  let str = '';
  _res.forEach((el) => {
    str += '<li>' + el + '</li>';
  });

  $('#viewer_datalist').empty().append(str);

  $('#viewer_datalist li')
    .off()
    .on('click', (e) => {
      $('#viewer_datalist li').removeClass('selected');
      $(e.currentTarget).addClass('selected');
      project = $(e.currentTarget).text();
      getImgList($(e.currentTarget).text());
    });
  $('#viewer_datalist li:first').click();
};

const getImgList = (projectName) => {
  const param = {
    userID: user.userID,
    folderName: projectName,
  };
  $.ajax({
    type: 'post',
    url: '/main/GetImageList',
    dataType: 'json',
    data: param,
    success: (res) => {
      if (res) {
        dataList = res.fileList;
        setImage(0);
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};

const setImage = (idx) => {
  $('#targetName').text(dataList[idx]);
  $('#imgCnt').text('(' + Number(idx + 1) + '/' + dataList.length + ')');
  $('#gt_img')[0].src =
    '../storage/output/bbox_plotted/gt/' + project + '/' + dataList[idx];
  $('#gt_hist')[0].src =
    '../storage/output/hist/gt/' +
    project +
    '/size_dist_' +
    dataList[idx].split('.JPG')[0] +
    '.png';
  $('#pred_hist')[0].src =
    '../storage/output/hist/pred/' +
    project +
    '/size_dist_' +
    dataList[idx].split('.JPG')[0] +
    '.png';
  $('#pred_img')[0].src =
    '../storage/output/bbox_plotted/pred/' + project + '/' + dataList[idx];
};

$('#gotoFirst').on('click', () => {
  setImage(0);
});

$('#gotoLast').on('click', () => {
  setImage(dataList.length - 1);
});

$('#gotoPre').on('click', () => {
  let index = Number($('#imgCnt').text().split('(')[1].split('/')[0]);
  if (index - 1 > 0) {
    setImage(index - 2);
  } else {
    setImage(0);
  }
});

$('#gotoNext').on('click', () => {
  let index = Number($('#imgCnt').text().split('(')[1].split('/')[0]);
  if (index != dataList.length) {
    setImage(index);
  }
});
