// 사용자 정보 없으면
if (!window.localStorage.getItem('user')) {
  location.href = '/login';
}

let user = JSON.parse(localStorage.getItem('user'));
let slide;
let uploadedFiles;
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

  $('#btnUpload').on('click', () => {
    $.blockUI({
      message: $('#pop_upload'),
      css: {
        width: '0px',
        border: 0,
      },
    });
  });

  $('#pop_dragArea').on('click', (e) => {
    e.preventDefault();
    $('#inputFile').click();
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
      setImage();
    });
  $('#viewer_datalist li:first').click();
};

const setImage = () => {
  let target = '';
  for (let i = 0; i < $('#viewer_datalist li').length; i++) {
    if ($($('#viewer_datalist li')[i]).hasClass('selected')) {
      target = $($('#viewer_datalist li')[i]).text();
      break;
    }
  }
  const param = {
    userID: user.userID,
    folderName: target,
  };
  $.ajax({
    type: 'post',
    url: '/main/GetImageList',
    dataType: 'json',
    data: param,
    success: (res) => {
      if (res) {
        console.log(res);
        if (res.fileList.length == 0) {
          alert('데이터가 없습니다. 이미지를 업로드해주세요.');
          return;
        }
        let str = '';
        $('#imgName').text(res.fileList[0]);

        str = '<div class="swiper-wrapper">';
        for (let i = 0; i < res.fileList.length; i++) {
          str += '<div class="swiper-slide mainImg">';
          str +=
            '<img src="/../storage/yolo_data/test/images/' +
            target +
            '/' +
            res.fileList[i] +
            '" />';
          str += '</div>';
        }
        str += '</div>';
        str += '<div class="swiper-pagination"></div>';
        str +=
          '<div class="swiper-button-prev btnSwiper" style="color: #4e944f;">';
        str += '<i class="fa-solid fa-circle-chevron-left fa-xl"></i></div>';
        str +=
          '<div class="swiper-button-next btnSwiper" style="color: #4e944f;">';
        str += '<i class="fa-solid fa-circle-chevron-right fa-xl"></i></div>';
        $('#imageArea').empty().append(str);

        slide = new Swiper('#imageArea', {
          //   autoplay: {
          //     delay: 5000,
          //   },
          loop: true,
          slidesPerView: 1,
          spaceBetween: 10,
          centeredSlides: true,
          pagination: {
            el: '#imageArea .swiper-pagination',
            clickable: true,
          },
          navigation: {
            prevEl: '#imageArea .swiper-button-prev',
            nextEl: '#imageArea .swiper-button-next',
          },
        });

        $('.btnSwiper').on('click', () => {
          // let idx = $('.swiper-slide-active')[0].getAttribute(
          //   'data-swiper-slide-index'
          // );
          // let img_name = $($('#viewer_datalist').find('li')[idx]).text();

          let img_src = $('.swiper-slide-active').find('img').attr('src');
          let img_name = img_src.split(target + '/')[1];
          $('#imgName').text(img_name);
        });
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};

const readInputFile = () => {
  const files = $('#inputFile')[0].files;
  checkFileList(files);
  uploadedFiles = files;
};

const $area = $('#pop_dragArea')[0];
if ($area) {
  $area.ondrop = (e) => {
    e.preventDefault();
    const files = [...e.dataTransfer?.files];
    checkFileList(files);
    uploadedFiles = files;
  };

  $area.ondragover = (e) => {
    e.preventDefault();
  };

  $area.ondragenter = (e) => {
    e.preventDefault();

    $area.classList.add('active');
  };

  $area.ondragleave = (e) => {
    e.preventDefault();

    $area.classList.remove('active');
  };
}

const checkFileList = (_fileList) => {
  if (_fileList.length == 0) {
    alert('파일이 없습니다.');
    return;
  }
  $('#pop_alarm').hide();
  let str = '';
  for (let i = 0; i < _fileList.length; i++) {
    str +=
      '<div class="row pop_list"><i class="fa-regular fa-image" style="padding-right: 5px;"></i><label>' +
      _fileList[i].name +
      '</label></div>';
  }
  $('#pop_UploadfileList').empty().append(str);
  $('#cntFile').text(_fileList.length);
};

$('#pop_upload_btnOk').on('click', () => {
  if (uploadedFiles.length == 0) {
    alert('업로드할 파일을 선택해주세요.');
    return;
  }

  if ($('#pop_newFolderName').val().trim() == '') {
    alert('프로젝트 이름을 입력해주세요.');
    return;
  }

  const formData = new FormData();
  formData.append('userID', user.userID);
  formData.append('projName', $('#pop_newFolderName').val());
  for (let i = 0; i < uploadedFiles.length; i++) {
    formData.append('file', uploadedFiles[i]);
  }

  $.ajax({
    type: 'post',
    url: '/main/uploadFile',
    enctype: 'multipart/form-data',
    processData: false,
    contentType: false,
    data: formData,
    timeout: 300000,
    // data: fileInfo,
    success: (res) => {
      if (res) {
        $.unblockUI();
        getDataList();
        // console.log(res);
      }
    },
    error: (req, status, err) => {
      $.unblockUI();
      console.log(err);
    },
  });
});

const setImage111 = (_data) => {
  let str = '<div class="swiper-wrapper">';
  for (let i = 0; i < _data.length; i++) {
    str += '<div class="swiper-slide mainImg">';
    str += '<img src="/../storage/yolo_data/test/images/' + _data[i] + '" />';
    str += '</div>';
  }
  str += '</div>';
  str += '<div class="swiper-pagination"></div>';
  str += '<div class="swiper-button-prev btnSwiper" style="color: #4e944f;">';
  str += '<i class="fa-solid fa-circle-chevron-left fa-xl"></i></div>';
  str += '<div class="swiper-button-next btnSwiper" style="color: #4e944f;">';
  str += '<i class="fa-solid fa-circle-chevron-right fa-xl"></i></div>';
  $('#imageArea').empty().append(str);

  slide = new Swiper('#imageArea', {
    //   autoplay: {
    //     delay: 5000,
    //   },
    loop: true,
    slidesPerView: 1,
    spaceBetween: 10,
    centeredSlides: true,
    pagination: {
      el: '#imageArea .swiper-pagination',
      clickable: true,
    },
    navigation: {
      prevEl: '#imageArea .swiper-button-prev',
      nextEl: '#imageArea .swiper-button-next',
    },
  });

  $('.btnSwiper').on('click', () => {
    let idx = $('.swiper-slide-active')[0].getAttribute(
      'data-swiper-slide-index'
    );
    let img_name = $($('#viewer_datalist').find('li')[idx]).text();
    $('#imgName').text(img_name);
  });
};
