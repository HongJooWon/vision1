$(document).ready(() => {
  const slide = new Swiper('#rawImg', {
    //   autoplay: {
    //     delay: 5000,
    //   },
    loop: true,
    slidesPerView: 1,
    spaceBetween: 10,
    centeredSlides: true,
    pagination: {
      el: '#rawImg .swiper-pagination',
      clickable: true,
    },
    navigation: {
      prevEl: '#rawImg .swiper-button-prev',
      nextEl: '#rawImg .swiper-button-next',
    },
  });

  $('#file').change(() => {
    let selectFile = $('#file')[0].files;

    const formData = new FormData();

    for (let i = 0; i < selectFile.length; i++) {
      formData.append('image', selectFile[i]);
    }
  });

  $('#btnModelStart').bind('click', () => {
    const param = {
      fileName: $('.upload-name').val(),
    };
    console.log(param);
    $.ajax({
      type: 'post',
      url: '/main/GetDetection',
      dataType: 'json',
      data: param,
      success: (res) => {
        if (res.tableData) {
          setTable(res.tableData);
          setImage(res.imgData);
        }
      },
      error: (req, status, err) => {
        console.log(err);
      },
    });

    console.log(`[CLICK] Model Start button`);
    // alert(`[CLICK] Model Start button`);
  });
});

const setImage = (_data) => {
  let str = '<div class="swiper-wrapper">';
  for (let i = 0; i < _data.length; i++) {
    str += '<div class="swiper-slide mainImg">';
    str += '<img src="' + _data[i] + '" />';
    str += '</div>';
  }
  str += '</div>';
  str += '<div class="swiper-pagination"></div>';
  str +=
    '<div class="swiper-button-prev btnSwiper" style="color: #4e944f; display:none;">';
  str += '<i class="fa-solid fa-circle-chevron-left fa-xl"></i></div>';
  str +=
    '<div class="swiper-button-next btnSwiper" style="color: #4e944f; display:none;">';
  str += '<i class="fa-solid fa-circle-chevron-right fa-xl"></i></div>';
  $('#detectedImg').empty().append(str);

  const slide = new Swiper('#detectedImg', {
    //   autoplay: {
    //     delay: 5000,
    //   },
    loop: true,
    slidesPerView: 1,
    spaceBetween: 10,
    centeredSlides: true,
    pagination: {
      el: '#detectedImg .swiper-pagination',
      clickable: true,
    },
    navigation: {
      prevEl: '#rawImg .swiper-button-prev',
      nextEl: '#rawImg .swiper-button-next',
    },
  });
};
