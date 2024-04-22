(function (Utils, $) {
    function renderButtonIconSVG(href, viewBox, state = '') {
        if (state) {
            state = ` class="${state}"`;
        }
        return `<svg viewBox="${viewBox}"${state}><use href="${href}"></use></svg>`;
    }

    Utils.renderButtonIcon = buttons => {
        buttons.each(function () {
            let $this = $(this);
            let icon = $this.attr("data-icon");
            let on = $this.attr("data-icon-on");
            let off = $this.attr("data-icon-off");
            let viewBox = $this.attr("data-icon-viewBox");
            if (typeof viewBox == "undefined") {
                viewBox = "0 0 256 256";
            }
            if (typeof off != "undefined") {
                $this.prepend(renderButtonIconSVG(off, viewBox, "off"));
            }
            if (typeof on != "undefined") {
                $this.prepend(renderButtonIconSVG(on, viewBox, "on"));
            }
            if (typeof icon != "undefined") {
                $this.prepend(renderButtonIconSVG(icon, viewBox));
            }
        });
    }
    Utils.timeFormat = (timestamp, filename = false) => {
        let date = new Date(timestamp);
        let year = date.getFullYear();
        let month = String(date.getMonth() + 1).padStart(2, '0');
        let day = String(date.getDate()).padStart(2, '0');
        let hour = String(date.getHours()).padStart(2, '0');
        let minute = String(date.getMinutes()).padStart(2, '0');
        let second = String(date.getSeconds()).padStart(2, '0');
        let millisecond = String(date.getMilliseconds()).padStart(3, '0');
        if (filename) {
            return `${year}-${month}-${day}_${hour}-${minute}-${second}`;
        }
        else {
            return `${year}-${month}-${day} ${hour}:${minute}:${second}.${millisecond}`;
        }
    }
    Utils.download = (content, fileName, contentType) => {
        let a = $(`<a href="${URL.createObjectURL(new Blob([content], { type: contentType }))}" download="${fileName}"></a>`).get(0);
        a.click();
        a.remove();
    }
    Utils.upload = (accept, onread) => {
        let input = $(`<input type="file" accept="${accept}">`).get(0);
        input.onchange = e => {
            onread(e.target.files);
        };
        input.click();
        input.remove();
    }

    $(document).ready(_ => {
        Utils.renderButtonIcon($(".icon"));
    });
}(window.Utils = window.Utils || {}, jQuery));