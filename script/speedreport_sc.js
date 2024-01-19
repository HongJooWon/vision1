// 사용자 정보 없으면
if (!window.localStorage.getItem('user')) {
  location.href = '/login';
}

let user = JSON.parse(localStorage.getItem('user'));

// #region [Datatables]
let exportFileName = 'Export Data';
let colList = [];
let table;
$(document).ready(() => {
  getBatchList();
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
  if (_res.length == 0) {
    alert('존재하는 배치가 없습니다.');
    return;
  }

  let str = '<option>배치 선택</option>';
  for (let i = 0; i < _res.length; i++) {
    str += '<option>' + _res[i] + '</option>';
  }

  $('#projList').empty().append(str);

  $('#projList').change((e) => {
    if ($(e.currentTarget).val() != '배치 선택') {
      getDrawData($(e.currentTarget).val());
    }
  });
};

const getDrawData = (batchName) => {
  if (batchName == '') {
    alert('선택된 배치가 없습니다.');
    return;
  }

  const param = {
    userID: user.userID,
    batchName: batchName,
  };

  $.ajax({
    type: 'post',
    url: '/speed/GetDrawData',
    dataType: 'json',
    data: param,
    traditional: true,
    success: (res) => {
      if (res) {
        setTable(res);
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};

const setTable = (_data) => {
  if (table) {
    table.destroy();
  }
  let value = _data.speed.toFixed(2) + ' Pixel/Seconds';
  $('#speed_txt').text(value);
  let col = [];
  for (let i = 0; i < _data.head.length; i++) {
    col.push({ title: _data.head[i] });
  }
  $('#reportTable').empty();

  table = $('#reportTable').DataTable({
    searching: false,
    dom: 'Bfrtip',
    buttons: [
      {
        extend: 'csvHtml5',
        title: function () {
          return exportFileName;
        },
      },
      {
        extend: 'pdfHtml5',
        title: function () {
          return exportFileName;
        },
      },
    ],
    // scrollX: true,
    // scrollCollapse: true,
    paging: true,
    fixedHeader: true,
    pageLength: 25,
    data: _data.data,
    columns: col,
  });

  $('.dt-buttons button').hide();

  // $('#reportTable_wrapper')[0].style.cssText = 'width: 100%; overflow-x:auto;';
  $('#btnCSV')
    .off()
    .on('click', () => {
      exportFileName = prompt('파일명을 입력하세요.', exportFileName);
      if (exportFileName == null) {
        exportFileName = 'Export Data';
        return;
      }
      $('.buttons-csv').click();
    });
  $('#btnPDF')
    .off()
    .on('click', () => {
      exportFileName = prompt('파일명을 입력하세요.', exportFileName);
      if (exportFileName == null) {
        exportFileName = 'Export Data';
        return;
      }
      $('.buttons-pdf').click();
    });
};
// #endregion
