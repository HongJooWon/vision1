// 사용자 정보 없으면
if (!window.localStorage.getItem('user')) {
  location.href = '/login';
}

let user = JSON.parse(localStorage.getItem('user'));
let slide;
let uploadedFiles;

$(document).ready(() => {
  $('#userName').text(user.userName);
  getBatchList();

  $('#searchWord').on('keyup', (e) => {
    let word = $('#searchWord').val();
    console.log(word);

    if (word == '') {
      $('#viewer_datalist li').show();
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

const getBatchList = () => {
  const param = {
    userID: user.userID,
  };

  $.ajax({
    type: 'post',
    url: '/speed/GetBatchList',
    dataType: 'json',
    data: param,
    success: (res) => {
      if (res) {
        setBatchList(res.fileList);
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};

const setBatchList = (_res) => {
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
      getSpeed($(e.currentTarget).text());
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
    url: '/speed/GetResultImage',
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

        str = '<div class="swiper-wrapper">';
        for (let i = 0; i < res.fileList.length; i++) {
          str += '<div class="swiper-slide mainImg">';
          str +=
            '<img src="/../storage/output/sentimentation/images/' +
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
          slidesPerView: 3,
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

const getSpeed = (batch_name) => {
  const param = {
    userID: user.userID,
    batchName: batch_name,
  };

  $.ajax({
    type: 'post',
    url: '/speed/GetSpeed',
    dataType: 'json',
    data: param,
    success: (res) => {
      if (res) {
        let value = res.speed.toFixed(2) + ' Pixel/Seconds';
        $('#speed_txt').text(value);
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};
