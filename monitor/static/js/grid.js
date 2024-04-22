(function (Grid, $) {
    let $rows = null;
    let heights = null;
    const observer = new ResizeObserver(resizeRows);

    function fetchHeight() {
        let height = [];
        for (let i = 0; i < heights.length; i++) {
            height.push($rows.eq(i).outerHeight());
        }
        return height;
    }
    function resizeRows() {
        let height = fetchHeight();
        let changed = null;
        for (let i = 0; i < height.length; i++) {
            if (heights[i] != height[i]) {
                changed = i;
                break;
            }
        }
        if (changed != null) {
            let top = 0;
            for (let i = 0; i < height.length; i++) {
                if (i > changed) {
                    $rows.eq(i).css('top', `${top}px`);
                }
                top += height[i];
            }
            $rows.last().css({
                top: `${top}px`,
                height: `calc(100vh - ${top}px)`
            });
            heights = height;
        }
    }

    $(document).ready(_ => {
        $rows = $('body > .grid-row');
        if ($rows.length > 1) {
            heights = Array.from({ length: $rows.length - 1 }, () => 0);
            resizeRows();
            $rows.each(function () {
                observer.observe(this);
            });
        }
    });
}(window.Grid = window.Grid || {}, jQuery));