(function (Index, $) {
    let $tab = null
    let $embed = null;

    function activeTab(tab) {
        if (!tab.hasClass('active')) {
            let jump = tab.next();
            let href = jump.attr('href');
            $('a[role="index-tab-title"]').removeClass('active');
            $('a[role="index-tab-button"]').removeClass('active');
            tab.addClass('active');
            jump.addClass('active');
            let embed = $embed.clone().attr('src', href);
            $embed.replaceWith(embed);
            $embed = embed;
        }
    }

    Index.switchTab = e => {
        activeTab($(e.target));
    }

    $(document).ready(_ => {
        $tab = $('#index-tab');
        $embed = $('#index-tab-content');
        activeTab($('a[role="index-tab-title"]').first());
    });
}(window.Index = window.Index || {}, jQuery));