$(document).ready(() => {
  setSelct();

  $('#selectImg').change(() => {
    const idx = parseInt($('#selectImg').val()) + 1;
    slide.slideTo(idx, 1000, false);
    changeInfo();
  });

  $('#btnUpload').bind('click', () => {
    $('#inputUpload').click();
  });

  $('#inputUpload').change(() => {
    let selectFile = $('#inputUpload')[0].files[0];
    console.log(selectFile);
    const file = URL.createObjectURL(selectFile);
    console.log(file);

    const formData = new FormData();
    formData.append('image', selectFile);

    for (let value of formData.values()) {
      console.log(value);
    }

    $.ajax({
      type: 'post',
      url: '/viewer/upload',
      //dataType: 'formData',
      data: formData,
      contentType: false,
      processData: false,
      success: (res) => {
        if (res.msg == 'success') {
          alert(`${selectFile.name}을 업로드하였습니다.`);
        }
      },
      error: (req, status, err) => {
        console.log(err);
      },
    });
  });
});

// #region [이미지 슬라이드]
const slide = new Swiper('#image-slide', {
  //   autoplay: {
  //     delay: 5000,
  //   },
  loop: true,
  slidesPerView: 1,
  spaceBetween: 10,
  centeredSlides: true,
  pagination: {
    el: '.swiper-pagination',
    clickable: true,
  },
  navigation: {
    prevEl: '.swiper-button-prev',
    nextEl: '.swiper-button-next',
  },
});

$('.btnSwiper').click(() => {
  let idx = $('.swiper-slide-active')[0].getAttribute(
    'data-swiper-slide-index'
  );
  changeInfo();
  $('#selectImg').val(idx);
});

const changeInfo = () => {
  let idx = $('.swiper-slide-active')[0].getAttribute(
    'data-swiper-slide-index'
  );
  let info = imgMetaData[parseInt(idx)];
  $('#imgID').text(info['imgID']);
  $('#fileName').text(info['fileName']);
  $('#picDate').text(info['picDate']);
  $('#location').text(info['location']);
  $('#picFormat').text(info['format']);
  $('#picSize').text(info['picSize']);
  $('#resolution').text(info['resolution']);
  $('#camera').text(info['camera']);
  $('#picAdmin').text(info['picAdmin']);
};
// #endregion

// #region [Data Selectbox 이벤트]
const setSelct = () => {
  let str = '';
  for (let i = 0; i < imgMetaData.length; i++) {
    str +=
      '<option value="' + i + '">' + imgMetaData[i]['dataName'] + '</option>';
  }
  $('#selectImg').empty().append(str);
};
// #endregion
