/***** GLOBAL VARIABLES *****/ 

var HOST_URL = 'http://localhost:5000'
var n_resultados = 10

/***** END GLOBAL VARIABLES *****/ 

/***** API FUNCTIONS *****/ 

function related_songs() {
	var search = $('#tema').val()
	console.log('results: ' + search)
	var predict = 'False'
	var send_data = {'search':search,
	          'predict':predict,
	          'cant_resultados':n_resultados,
	          'instrumentacion':$('#instrumentacion').val(),
	          'ritmica_drums':$('#ritmica_drums').val(),
	          'ritmica_instrument':$('#ritmica_instrument').val(),
	          'amplitud_tonal':$('#amplitud_tonal').val(),
	          'dinamica':$('#dinamica').val(), 
	          'duracion_notas':$('#duracion_notas').val(), 
	          'notas_simultaneas':$('#notas_simultaneas').val(),
	          'tempo':$('#tempo').val(),
	          'duracion_tema':$('#duracion_tema').val(),
	          'armonia':$('#armonia').val(),
	          'others':$('#others').val(),
	          'penalizacion':$('#penalizacion').val(),
	          'filtrar':$('#filtrar').prop('checked'),
	          'clusterizar':$('#clusterizar').prop('checked'),
	          'page_number':parseInt($('#pager .page')[0].innerHTML) - 1}
     
	var $ul = $("#resultados");
	$ul.append('<div class="loader"></div>');
    $.ajax({
        type        : 'GET',
        url         : HOST_URL + '/related_songs',
       	data 		: send_data,
        dataType    : 'json',
        success: function(data) {
            console.log(Object.values(data).sort()); 
	        $ul.empty();

			for (i = 0; i < n_resultados; i++) {
				key = data.path[i]
				value = data.value[i]
		      	$ul.append('<li class="list-group-item results"><p class="key">' + key + '</p><span class="value">(' + Math.round(parseFloat(value) * 1000) / 1000 + ')</span><span class="play_song" data-key="' + key + '">Play</span><a src="' + HOST_URL + '/get-file/' + key + '"><span class="download_song" data-key="' + key + '">download</span></a><span class="search_song" data-key="' + key + '">Search</span></li>');
			}
		},
	    complete: function (data) {
			$('#pager').show(100);
		}
	});
}

function search_titles() {
	var search = $('#tema').val()
	$('form').hide(300);
	var $ul = $("#resultados");
	$('#visualizer_container').hide(100);
    if($("#tema").val().length > 2) {
		$('#resultados').removeClass('large')
    	console.log('searching key:' + search)
		$.ajax({
		        type        : 'GET',
		        url         : HOST_URL + '/search_titles',
		       	data 		: {'search': $("#tema").val(),
				   			   'cant_resultados':n_resultados,
				   			   'page_number':parseInt($('#pager .page')[0].innerHTML) - 1},
		        dataType    : 'json',
		        success: function(data) {
		            console.log(Object.values(data).sort()); 
					$ul.empty();
					for (var key in data) {
				      	$ul.append('<li class="list-group-item search" data-key="' + key + '"><span class="key">' + key + '</span></li>');
					}			
				},
			    complete: function (data) {
					$('#pager').show(100);
				}
		});
	} else {
		$ul.empty();
	}
}
/***** END API FUNCTIONS *****/ 

/***** ON CLICK FUNCTIONS *****/ 

$(document).on('click','header .w3-button', function() {
	$("#tema").val('');
	$("#tema").focus();
});

$(document).on('click','#resultados li.results span.play_song', function() {
	  $('#resultados li.results').removeClass('active')
	  $(this).parent('li').addClass('active')
	   var tema = $(this).attr('data-key');
	   var url_tema = HOST_URL + '/get-file/' + tema
	   $('#midi_player').attr('src',url_tema)
	   $('#listening')[0].innerHTML = 'Listening ' + tema
	   $('#visualizer_container').show();
});

$(document).on('click','#resultados li.results span.search_song', function() {
	var tema = $(this).attr('data-key');
		$("#tema").val(tema);
		related_songs();
});

$(document).on('click','#hide_visualizer_btn', function() {
	if ($('#visualizer_container').hasClass('hidden')) {
		$('#visualizer_container').removeClass('hidden');
		$('#myPianoRollVisualizer').show(300);
	} else {
		$('#visualizer_container').addClass('hidden');
		$('#myPianoRollVisualizer').hide(300);
	}
});

$(document).on('click','#resultados li.search', function() {
	$('#pager .page')[0].innerHTML = 1
	var tema = $(this).attr('data-key');
	$("#tema").val(tema);
	related_songs();
	$('form').show(300);
	setTimeout(function(){
		$('#resultados').addClass('large')
	},300);		   		
});

$(document).on('change','form input', function() {
	$('#pager .page')[0].innerHTML = 1
	related_songs() 
});

function pageDown() {
    var pager = $('#pager .page')
    var pager_value = $(pager)[0].innerHTML
    if (pager_value > 1){
        $(pager)[0].innerHTML = parseInt(pager_value) - 1
		if ($('#resultados li.search').length > 0) {
			search_titles()
		} else {
			related_songs()
		}
    }
}

function pageUp() {
    $('#pager .page')[0].innerHTML = parseInt($('#pager .page')[0].innerHTML) + 1
	if ($('#resultados li.search').length > 0) {
		search_titles()
	} else {
		related_songs()
	}
}

/***** END ON CLICK FUNCTIONS *****/ 

$("#tema").keyup(function(e) {
	$('#pager .page')[0].innerHTML = 1
	$('#visualizer_container').hide(100);
    search_titles(); 
});

$('#tema').on('focus', function () {
    this.select();
});