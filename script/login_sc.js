window.localStorage.clear();

$('#userID').on('keydown', () => {
  $('.errID').hide();
});

$('#password').on('keydown', () => {
  $('.errPa').hide();
});

$('#loginBtn').on('click', () => {
  login();
});

const tryLogin = () => {
  if (window.event.keyCode == 13) {
    login();
  }
};

const login = () => {
  event.preventDefault();
  const userID = $('#userID').val() == '' ? 'admin' : $('#userID').val();
  const password = $('#password').val() == '' ? 'attic1' : $('#password').val();

  if (userID.trim() == '') {
    $('.errID').show();
    return;
  }
  if (password.trim() == '') {
    $('.errPa').show();
    return;
  }

  const param = {
    userID: userID,
    password: password,
  };

  $.ajax({
    type: 'post',
    url: '/login/Signin',
    dataType: 'json',
    data: param,
    success: (res) => {
      if (res) {
        if (res.Status == '001') {
          alert('ID를 확인해주세요.');
          $('#userID').focus();
          return;
        } else if (res.Status == '002') {
          alert('Password를 확인해주세요.');
          $('#password').focus();
          return;
        } else if (res.Status == '000') {
          const user = {
            userID: res.userID,
            userName: res.userName,
            token: res.token,
          };
          localStorage.setItem('user', JSON.stringify(user));
          location.href = '/main';
        } else {
          console.log(res);
        }
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};
