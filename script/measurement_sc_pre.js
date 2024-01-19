const mainSlide = new Swiper('#detectedImg', {
  loop: true,
  slidesPerView: 1,
  spaceBetween: 10,
  centeredSlides: true,
  pagination: {
    el: '#detectedImg .swiper-pagination',
    clickable: true,
  },
  navigation: {
    prevEl: '#detectedImg .swiper-button-prev',
    nextEl: '#detectedImg .swiper-button-next',
  },
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
    el: '#image-slide .swiper-pagination',
    clickable: true,
  },
  navigation: {
    prevEl: '#image-slide .swiper-button-prev',
    nextEl: '#image-slide .swiper-button-next',
  },
});

$('.btnSwiper').click(() => {
  let idx = $('.swiper-slide-active')[0].getAttribute(
    'data-swiper-slide-index'
  );
  changeInfo();
});

const changeInfo = () => {
  let idx = $('.mainImg.swiper-slide-active')[0].getAttribute(
    'data-swiper-slide-index'
  );
  let info = imgMetaData[parseInt(idx)];
  $('#imgID').text(info['fileName']);
  $('#length').text(info['length'] + ' mm');
  $('#width').text(info['width'] + ' mm');

  let str = '<div class="swiper-wrapper">';
  for (let i = 0; i < info['edge'].length; i++) {
    str += '<div class="swiper-slide twice">';
    str += '<img src="../data/images' + info['edge'][i] + '" />';
    str += '</div>';
  }
  str += '</div>';
  str += '<div class="swiper-pagination" id="paginationSub"></div>';
  str +=
    '<div class="swiper-button-prev" id="prevSub" style="color: #4e944f;"><i class="fa-solid fa-circle-chevron-left fa-xl"></i></div>';
  str +=
    '<div class="swiper-button-next" id="nextSub" style="color: #4e944f;"><i class="fa-solid fa-circle-chevron-right fa-xl"></i></div>';
  $('#image-slide').empty().append(str);

  const newslide = new Swiper('#image-slide', {
    loop: true,
    slidesPerView: 1,
    spaceBetween: 10,
    centeredSlides: true,
    pagination: {
      el: '#image-slide .swiper-pagination',
      clickable: true,
    },
    navigation: {
      prevEl: '#image-slide .swiper-button-prev',
      nextEl: '#image-slide .swiper-button-next',
    },
  });

  //findEdge();
};

// const findEdge = () => {
//   let idx = $('.mainImg.swiper-slide-active')[0].getAttribute(
//     'data-swiper-slide-index'
//   );
//   let subidx = $('#image-slide')
//     .find('.swiper-slide-active')[0]
//     .getAttribute('data-swiper-slide-index');
//   let info = imgMetaData[parseInt(idx)];
//   const edgeName = info['edge'][subidx].split('edge/')[1];
//   $('#edgeID').text(edgeName);
// };
// #endregion

// #region [버튼 이벤트 관리]
$('#btnStart').click(() => {
  console.log('click start button');
});
$('#btnStop').click(() => {
  console.log('click stop button');
});
$('#btnCancel').click(() => {
  console.log('click cancel button');
});
// #endregion
