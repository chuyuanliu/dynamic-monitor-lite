(function (Dashboard, $) {
    function activeNav(selected) {
        if (!selected.hasClass('active')) {
            $('a[role="dashboard-nav-button"]').removeClass('active');
            selected.addClass('active');
        }
    }
    function activeNavByHref(href) {
        activeNav($(`a[role="dashboard-nav-button"][href="${href}"]`));
    }

    Dashboard.switchNavByHref = e => {
        activeNavByHref($(e.target).attr('href'));
    }
    Dashboard.switchNav = e => {
        activeNav($(e.target));
    }

    $(document).ready(_ => {
        activeNavByHref(window.location.hash);
    });
}(window.Dashboard = window.Dashboard || {}, jQuery));