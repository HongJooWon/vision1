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
  getProjectList();
});

const getProjectList = () => {
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
        setProjectList(res.fileList);
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};

const setProjectList = (_res) => {
  if (_res.length == 0) {
    alert('존재하는 프로젝트가 없습니다.');
    return;
  }

  let str = '<option>--- 프로젝트 선택 ---</option>';
  for (let i = 0; i < _res.length; i++) {
    str += '<option>' + _res[i] + '</option>';
  }

  $('#projList').empty().append(str);

  $('#projList').change((e) => {
    if ($(e.currentTarget).val() != '--- 프로젝트 선택 ---') {
      getColList($(e.currentTarget).val());
    }
  });
};

const getColList = (projName) => {
  const param = {
    userID: user.userID,
    projName: projName,
  };

  $.ajax({
    type: 'post',
    url: '/report/GetColList',
    dataType: 'json',
    data: param,
    success: (res) => {
      if (res) {
        console.log(res);
        setColList(res.head);
      }
    },
    error: (req, status, err) => {
      console.log(err);
    },
  });
};

const setColList = (_res) => {
  if (_res.length == 0) {
    alert('데이터가 없습니다.');
    return;
  }

  let str =
    '<label><input type="checkbox" name="colName" value="selectAll" onclick="selectAll(this)" />전체 선택</label>';
  // let str = '';
  for (let i = 0; i < _res.length; i++) {
    str +=
      '<label style="cursor: pointer;"><input type="checkbox" name="colName" value="' +
      _res[i] +
      '" />' +
      _res[i] +
      '</label>';
  }

  $('#imgList').empty().append(str);

  $('input[type=checkbox][name=colName]').change(function () {
    if ($(this).is(':checked')) {
      if (this.value == 'selectAll') {
        return;
      }
      colList.push(this.value);
      getDrawData();
    } else {
      if (this.value == 'selectAll') {
        return;
      }
      colList = colList.filter((target) => target !== this.value);
      getDrawData();
    }
  });
};

const selectAll = (target) => {
  const checkboxs = document.getElementsByName('colName');
  checkboxs.forEach((el) => {
    if (target.checked) {
      colList.push(el.value);
    } else {
      colList = colList.filter((target) => target !== el.value);
    }
    el.checked = target.checked;
  });
  getDrawData();
};

const getDrawData = () => {
  if (colList.length == 0) {
    alert('선택된 데이터가 없습니다.');
    $('#reportTable').empty();
    return;
  }

  const param = {
    userID: user.userID,
    projName: $('#projList').val(),
    columnList: colList,
    colCNT: colList.length,
  };

  $.ajax({
    type: 'post',
    url: '/report/GetDrawData',
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
  console.log(_data);
  if (table) {
    table.destroy();
  }

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
